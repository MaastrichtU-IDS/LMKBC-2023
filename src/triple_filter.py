import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import json
# import jsonlines

import heapq

import util
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity
import transformers as t
import config

import os
print("cwd", os.getcwd())
import evaluate
import pandas as pd
from tqdm import tqdm

entity_embedding_dict = dict()
def generate_triples_from_jsonl(file_path):
    triples = []
    json_lines = util.file_read_json_line(file_path)
    for line in json_lines:
        subject_entity_id = line["SubjectEntity"]
        relation = line["Relation"]
        object_entities = line["ObjectEntities"]

        # Create triples for each object entity (if any)
        for object_entity in object_entities:
          if object_entity =="":
            object_entity = 'No_'+relation
          triples.append((subject_entity_id, relation, object_entity))

    return triples

# Example usage:



def transform_to_tensors(triples, entity_id_to_idx, relation_to_idx):
    # Create lists to store the indices of entities and relations
    head_entities_idx = []
    relations_idx = []
    tail_entities_idx = []
    triples_v =[]

    for triple in triples:
        head_entity, relation, tail_entity = triple

        # Convert entity and relation strings to their respective indices
        head_entity_idx = entity_id_to_idx[head_entity]
        relation_idx = relation_to_idx[relation]
        tail_entity_idx = entity_id_to_idx[tail_entity]

        triples_v.append([head_entity_idx, relation_idx, tail_entity_idx])

    # Convert lists to PyTorch tensors
    triple_idx = torch.tensor(triples_v, dtype=torch.long)


    return triple_idx


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin

        # Initialize entity and relation embeddings randomly
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

    def distance(self, head_entity, relation, tail_entity):
        # Scoring function (L1 distance)
        return torch.norm(head_entity + relation - tail_entity, p=1, dim=1)

    def _get_positive_sample(self, triples):
        # Randomly choose a positive triple
        return triples[torch.randint(0, triples.shape[0], (1,))]

    def _get_negative_sample(self, triples):
        # Randomly choose a negative triple with a corrupted tail entity
        corrupted_triples = triples.clone()
        corrupted_triples[:, 2] = torch.randint(0, self.num_entities, (corrupted_triples.shape[0],))
        return self._get_positive_sample(corrupted_triples)

    def forward(self, pos_head, pos_rel, pos_tail, neg_head, neg_rel, neg_tail):

        self.entity_embeddings.weight.data[:-1,:].div_(self.entity_embeddings.weight.data[:-1,:].norm(p=2,dim=1,keepdim=True))
        pos_distance = self.distance(self.entity_embeddings(pos_head), self.relation_embeddings(pos_rel), self.entity_embeddings(pos_tail))
        neg_distance = self.distance(self.entity_embeddings(neg_head), self.relation_embeddings(neg_rel), self.entity_embeddings(neg_tail))
        return pos_distance, neg_distance



def get_distance(s_label, r_label, o_label):
  s= entity_to_idx[s_label]
  o= entity_to_idx[o_label]
  r=relation_to_idx[r_label]
  return transE_model.distance(entity_embeddings[[s]], relation_embeddings[[r]], entity_embeddings[[o]])

def get_entity_embedding(e_label):
  e= entity_to_idx[e_label]
  return entity_embeddings[e]

def find_most_similar_entities(query_embedding, entity_embeddings, k=5):
    """
    Find the k most similar entities for a given query embedding among a list of entity embeddings.

    Parameters:
        query_embedding (torch.Tensor): The embedding of the query entity.
        entity_embeddings (torch.Tensor): The embeddings of all entities in the knowledge graph.
        k (int): The number of most similar entities to return.

    Returns:
        list: A list of indices of the k most similar entities in the list of entity embeddings.
    """
    # Calculate cosine similarity between the query embedding and all entity embeddings
    similarity_scores = cosine_similarity(query_embedding.unsqueeze(0), entity_embeddings)

    # Get the indices of the k entities with highest similarity scores
    most_similar_indices = np.argsort(similarity_scores[0])[::-1][:k]

    return [idx_to_entity[e] for e in most_similar_indices]

# Example usage:
# Assuming you have a query entity embedding 'query_embedding' and a tensor containing all entity embeddings 'entity_embeddings'

# The indices in 'most_similar_indices' correspond to the k most similar entities in 'entity_embeddings'
# You can access the embeddings of the most similar entities using 'entity_embeddings[most_similar_indices]'

def map_kg_space(entity, tokenizer, model,centroid_bert, centroid_word2vec):
    if entity in entity_embedding_dict:
        word_bert = entity_embedding_dict[entity]
    else:
        with torch.no_grad():
            bert_model.eval()
            #print  (entity)
            inputs = tokenizer.encode(entity, add_special_tokens=True, return_tensors='pt')
            outputs = bert_model(inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            word_bert = embedding[0]
            entity_embedding_dict[entity]=word_bert
    aligned_w2v =(word_bert - centroid_bert)@ rotation_matrix
    aligned_w2v += centroid_word2vec
    return aligned_w2v
    

def find_most_similar_entities_in_kg(entity, tokenizer, model,centroid_bert, centroid_word2vec, entity_embeddings):
    aligned_w2v = map_kg_space(entity, tokenizer, model,centroid_bert, centroid_word2vec)
    return find_most_similar_entities(aligned_w2v, entity_embeddings, 5)
  

def ranking_prompt_output_batch(outputs, input_rows, bias):
    result_list=[]
    for row, output in zip(input_rows, outputs):
        #print (o)
        relation = row[config.KEY_REL]
        subject = row[config.KEY_SUB]
        relation_id = relation_to_idx[relation]
        thres = thresholds_rel[relation]
        # objects  = []
        ranks ={}
        for seq in output:
            obj_entity= seq['token_str']
            aligned_s = map_kg_space(subject, tokenizer, model, centroid_bert, centroid_transe)
            aligned_r = relation_embeddings[relation_id]
            aligned_o = map_kg_space(obj_entity, tokenizer, model, centroid_bert, centroid_transe)
            #print (aligned_s.shape, aligned_r.shape, aligned_o.shape)
            score = transE_model.distance(aligned_s.view(1, 768), aligned_r.view(1, 768), aligned_o.view(1, 768))[0]
            #if score <= 610.4186:
            if score <= thres + bias:
                ranks[obj_entity]= score
        # ranks = {outputs[i]['token_str']: scores[i].item() for i in range(len(outputs)) if scores[i] <= thres + bias}
        top_10_results = heapq.nsmallest(10, ranks, key=ranks.get)
        result_row = {
                "SubjectEntityID": row["SubjectEntityID"],
                "SubjectEntity": subject,
                # "ObjectEntitiesID": objects_wikiid,
                "ObjectEntities": top_10_results,
                "Relation": relation,
            }
        result_list.append(result_row)

    return result_list
  

def ranking_prompt_output(outputs, subj, relation_embeddings, relation_id, thres= 620.4186): #tokenizer, model, centroid_bert, centroid_transe, transE_model
    aligned_s = map_kg_space(subj, tokenizer, model, centroid_bert, centroid_transe)
    aligned_r = relation_embeddings[relation_id]

    aligned_o_list = []
    for o in outputs:
        obj_entity = o['token_str']
        aligned_o = map_kg_space(obj_entity, tokenizer, model, centroid_bert, centroid_transe)
        aligned_o_list.append(aligned_o)

    # Combine the embeddings of all object entities into a single tensor
    aligned_o_combined = torch.stack(aligned_o_list)

    # Resize aligned_s to match the dimensions of aligned_o_combined
    aligned_s = aligned_s.unsqueeze(0).expand(aligned_o_combined.size(0), -1)

    # Compute the distances between aligned_s, aligned_r, and all aligned_o entities using the transE_model
    scores = transE_model.distance(aligned_s, aligned_r.view(1, 768), aligned_o_combined)

    # Create a dictionary to store entity scores
    ranks = {outputs[i]['token_str']: scores[i].item() for i in range(len(outputs)) if scores[i] <= thres}

    top_10_results = heapq.nsmallest(10, ranks, key=ranks.get)
    return top_10_results

def train_transE(data_loader,transE_model,num_epochs =500):
    # Create DataLoader for batches of triples

    
    # Initialize the TransE model
 
    # Define the optimizer and loss function
    optimizer = optim.SGD(transE_model.parameters(), lr=0.01)
    loss_function = nn.MarginRankingLoss(margin=margin)
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        losses = []
        for batch_triples in data_loader:
            pos_head, pos_rel, pos_tail = batch_triples[:, 0], batch_triples[:, 1], batch_triples[:, 2]
            neg_triples = transE_model._get_negative_sample(batch_triples)
            neg_head, neg_rel, neg_tail = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]

            # Forward pass
            pos_distance, neg_distance = transE_model(pos_head, pos_rel, pos_tail, neg_head, neg_rel, neg_tail)

            # Compute the loss and backpropagate
            loss = loss_function(pos_distance, neg_distance, -1*torch.ones_like(pos_distance))
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            if len(losses)!=0:
                  mean_loss = np.array(losses).mean()
                  print('epoch {},\t train loss {:0.02f}'.format(epoch,mean_loss))
            else:
                  mean_loss = 0

    return transE_model

if __name__ == '__main__':
    file_path = "/content/LMKBC-2023/data/train.jsonl"
    triples = generate_triples_from_jsonl(config.TRAIN_FN)
    print(len(triples))
    model_name = 'bert-base-uncased'  # You can choose other variants as well
    tokenizer:t.BertTokenizer = t.BertTokenizer.from_pretrained(model_name)
    bert_model:t.BertModel = t.BertModel.from_pretrained(model_name)
      # Create entity and relation mapping to indices
    entities = set([triple[0] for triple in triples] + [triple[2] for triple in triples])
    relations = set([triple[1] for triple in triples])
    entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}
    relation_to_idx = {relation: idx for idx, relation in enumerate(relations)}

    # Transform triples into torch tensors
    triple_idx = transform_to_tensors(triples, entity_to_idx, relation_to_idx)

    input = config.VAL_FN
    print (f"Loading the input file \"{input}\"...")
    input_rows = [json.loads(line) for line in open(input, "r")]
    print (f"Loaded {len(input_rows):,} rows.")

      # Example data
    num_entities = len(entity_to_idx)
    num_relations = len(relation_to_idx)
    embedding_dim = 768
    margin = 1.0

    batch_size = 50
    data_loader = torch.utils.data.DataLoader(triple_idx, batch_size=batch_size, shuffle=True)

    transE_model = TransE(num_entities, num_relations, embedding_dim=embedding_dim, margin=margin)

    trans_fp = 'bin/transE'
    if os.path.exists(trans_fp):
        transE_model = torch.load(trans_fp)
    else:
        transE_model = train_transE(data_loader,transE_model,num_epochs=500)
        torch.save(transE_model, trans_fp)
 
    #  prompts.append(prompt)
    # print (prompt)
    batch_size = 2
    model_type = 'bert-base-uncased'
    tokenizer = t.AutoTokenizer.from_pretrained(model_type)
    model = t.AutoModelForMaskedLM.from_pretrained(model_type)  if "bert" in model_type.lower()  else t.AutoModelForCausalLM.from_pretrained(model_type)
    task = "fill-mask" if "bert" in model_type.lower() else "text-generation"


    pipe = t.pipeline(task=task, model=model, tokenizer=tokenizer, top_k=50)


        # Get the learned embeddings
    entity_embeddings = transE_model.entity_embeddings.weight.data
    relation_embeddings = transE_model.relation_embeddings.weight.data

    thresholds_rel = {}
    for r in relation_to_idx.keys():
      # print (r)
      scores = []
      for t in triples:
        if t[1] == r:
          scores.append(get_distance(t[0], t[1], t[2]).item())
    thresholds_rel[r] = np.mean(scores, axis=0)


    # Print learned embeddings
    # print("Entity embeddings:")
    # print(entity_embeddings)
    # print("Relation embeddings:")
    # print(relation_embeddings)

      
    thresholds_rel = {}
    for r in relation_to_idx.keys():
      # print (r)
      scores = []
      for t in triples:
        if t[1] == r:
          scores.append(get_distance(t[0], t[1], t[2]).item())
      thresholds_rel[r] = np.mean(scores, axis=0)


    bert_embeddings = []
    with torch.no_grad():
        bert_model.eval()
        for entity in entity_to_idx.keys():
            #print  (entity)
            inputs = tokenizer.encode(entity, add_special_tokens=True, return_tensors='pt')
            outputs = bert_model(inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
            #print(embedding)
            bert_embeddings.append(embedding)

    bert_embeddings =torch.tensor(bert_embeddings)
    centroid_transe = torch.mean(entity_embeddings, axis=0)

    centroid_bert = torch.mean(bert_embeddings, axis=0)
    centered_bert = bert_embeddings - centroid_bert
    centered_transe = entity_embeddings - centroid_transe
    rotation_matrix, _ =  orthogonal_procrustes(centered_bert, centered_transe)

    idx_to_entity = {v: k for k, v in entity_to_idx.items()}

    # Find the 5 most similar entities
    k = 5
    most_similar_indices = find_most_similar_entities(get_entity_embedding('Russia'), entity_embeddings, k)
    prompt_templates = util.file_read_prompt(config.prompt_fp)

    # Create prompts
    prompts = [ ]
    preds = []
    for row in input_rows:
        relation =row['Relation']
        subject_entity = row['SubjectEntity']
        prompt_template = prompt_templates[relation]
        prompt = prompt_template.format(subject_entity=subject_entity, mask_token=tokenizer.mask_token)
        #break
        prompts.append(prompt)
        # print (prompt)
    batch_size = 2
    outputs = pipe(prompts, batch_size=batch_size)
    # results = util.assemble_result(input_rows, outputs)

    results = ranking_prompt_output_batch(outputs, input_rows,  bias = 50)

    metrics = evaluate.evaluate_list(input_rows, results)

    scores_per_relation_pd = pd.DataFrame(metrics)
    print(scores_per_relation_pd.transpose().round(2))
    # Create prompts
    # prompts = [ ]
    # preds = []
    # for row in input_rows:
    #     relation =row['Relation']
    #     subject_entity = row['SubjectEntity']
    #     prompt_template = prompt_templates[relation]
    #     prompt = prompt_template.format(subject_entity=subject_entity, mask_token=tokenizer.mask_token)

    # outputs = pipe(prompts, batch_size=batch_size)
    # results = ranking_prompt_output(outputs, subject_entity, relation_embeddings, relation_to_idx[relation], thresholds_rel[relation]-10)
    # exc
    # # print (results, row['ObjectEntities'])
    # row['ObjectEntities_pred'] = results
    # preds.append(row)






  