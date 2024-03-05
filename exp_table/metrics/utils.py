import itertools
import torch
from sentence_transformers import SentenceTransformer
import evaluate
import pandas as pd
import numpy as np
import networkx as nx
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint


def KG_init():
    print("...initializing KG")
    print("...start loading the KG")
    semmed_db = pd.read_csv(
        "PREDICATIONS_OCCURS.csv",
        usecols=["SUBJECT_NAME", "OBJECT_NAME", "PREDICATE", "OCCURS"],
    )
    print("...finished loading the KG")

    semmed_db["SUBJECT_NAME"] = semmed_db["SUBJECT_NAME"].apply(lambda x: x.upper())
    semmed_db["OBJECT_NAME"] = semmed_db["OBJECT_NAME"].apply(lambda x: x.upper())
    G = nx.from_pandas_edgelist(
        semmed_db,
        source="SUBJECT_NAME",
        target="OBJECT_NAME",
        edge_attr="OCCURS",
        edge_key="PREDICATE",
    )
    concepts = set(nx.nodes(G))
    return G, concepts


def BERT_init():
    print("...initializing BERT")
    clinicalBert_tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    clinicalBert_model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
    return clinicalBert_tokenizer, clinicalBert_model


def get_model(model_name):
    if model_name == "GPT-3.5":
        print(
            "You need to get the responses separately from https://chat.lmsys.org/ to avoid API timeout issues."
        )
        return
    if model_name == "medalpaca":
        medalpaca_model = pipeline(
            "text-generation",
            model="medalpaca/medalpaca-7b",
            tokenizer="medalpaca/medalpaca-7b",
        )
        return medalpaca_model


def get_dataset(dataset_name):
    if dataset_name == "medmcqa":
        print("Currently only supports use of validation set")
        medmcqa_data = load_dataset("medmcqa", split="validation")
        return medmcqa_data


def get_responses(model, prompt_type, dataset, num_questions=20, max_new_tokens=128):
    """
    model: Huggingface model/pipeline
    prompt_type: str - can be "zero shot" or "zero shot with cot", "zero shot with template and cot", "1 shot with template and cot"
    Returns responses for first 20 questions in the dataset
    """
    QnA_dict = {}
    if prompt_type == "zero shot":
        for i in range(num_questions):
            print("Question: ", dataset["question"][i])
            input = f"""
            Question: {dataset["question"][i]}
            Option A: {dataset["opa"][i]}
            Option B: {dataset["opb"][i]}
            Option C: {dataset["opc"][i]}
            Option D: {dataset["opd"][i]}
            Answer:
            """
            generated_answer = model(input, max_new_tokens=128)
            generated_cot = generated_answer[0]["generated_text"][len(input) :]
            cot_gt = dataset["exp"][i]
            QnA_dict[dataset["question"][i]] = {
                "generated_cot": generated_cot,
                "cot_gt": cot_gt,
                "opa": dataset["opa"][i],
                "opb": dataset["opb"][i],
                "opc": dataset["opc"][i],
                "opd": dataset["opd"][i],
                "cop": dataset["cop"][i],
            }
    elif prompt_type == "zero shot with cot":
        for i in range(num_questions):
            print("Question: ", dataset["question"][i])
            input = f"""
            Question: {dataset["question"][i]}
            Option A: {dataset["opa"][i]}
            Option B: {dataset["opb"][i]}
            Option C: {dataset["opc"][i]}
            Option D: {dataset["opd"][i]}
            Answer: Let's think step by step.
            """
            generated_answer = model(input, max_new_tokens=128)
            generated_cot = generated_answer[0]["generated_text"][len(input) :]
            cot_gt = dataset["exp"][i]
            QnA_dict[dataset["question"][i]] = {
                "generated_cot": generated_cot,
                "cot_gt": cot_gt,
                "opa": dataset["opa"][i],
                "opb": dataset["opb"][i],
                "opc": dataset["opc"][i],
                "opd": dataset["opd"][i],
                "cop": dataset["cop"][i],
            }
    elif prompt_type == "zero shot with template and cot":
        for i in range(num_questions):
            print("Question: ", dataset["question"][i])
            input = f"""
            1. <step1>
            2. <step2>
            ...
            So the answer is (<answer>).
            Make sure that the answer uses the above format and answers the question step by step.
            Question: {dataset["question"][i]}
            Option A: {dataset["opa"][i]}
            Option B: {dataset["opb"][i]}
            Option C: {dataset["opc"][i]}
            Option D: {dataset["opd"][i]}
            Answer:
            """
            generated_answer = model(input, max_new_tokens=128)
            generated_cot = generated_answer[0]["generated_text"][len(input) :]
            cot_gt = dataset["exp"][i]
            QnA_dict[dataset["question"][i]] = {
                "generated_cot": generated_cot,
                "cot_gt": cot_gt,
                "opa": dataset["opa"][i],
                "opb": dataset["opb"][i],
                "opc": dataset["opc"][i],
                "opd": dataset["opd"][i],
                "cop": dataset["cop"][i],
            }
    elif prompt_type == "1 shot with template and cot":
        for i in range(num_questions):
            print("Question: ", dataset["question"][i])
            input = f"""
            1. <step1>
            2. <step2>
            ...
            So the answer is (<answer>).
            Make sure that the answer uses the above format. Answer the question step by step.

            Question: Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma:
            A) Hyperplasia
            B) Hyperophy
            C) Atrophy
            D) Dyplasia
            1. Chronic urethral obstruction because of urinary calculi, prostatic hyperophy, tumors, normal pregnancy, tumors, uterine prolapse or functional disorders cause hydronephrosis.
            2. Hydronephrosis is the dilatation of renal pelvis and calculus associated with progressive atrophy of the kidney due to obstruction to the outflow of urine.
            So the answer is C.

            Question: {dataset["question"][i]}
            Option A: {dataset["opa"][i]}
            Option B: {dataset["opb"][i]}
            Option C: {dataset["opc"][i]}
            Option D: {dataset["opd"][i]}
            """
            generated_answer = model(input, max_new_tokens=128)
            generated_cot = generated_answer[0]["generated_text"][len(input) :]
            cot_gt = dataset["exp"][i]
            QnA_dict[dataset["question"][i]] = {
                "generated_cot": generated_cot,
                "cot_gt": cot_gt,
                "opa": dataset["opa"][i],
                "opb": dataset["opb"][i],
                "opc": dataset["opc"][i],
                "opd": dataset["opd"][i],
                "cop": dataset["cop"][i],
            }
    return QnA_dict


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def get_paths(key_phrases, G):
    # only if more than 2 concepts are present in the KG
    if len(key_phrases) < 2:
        print(
            "Cannot perform this analysis without atleast 2 concepts present in the KG"
        )
        return
    else:
        # find shortest paths that exist between the concepts
        combinations_of_concepts = itertools.combinations(key_phrases, 2)
        paths = []
        for combination in combinations_of_concepts:
            shortest_paths = list(
                nx.all_shortest_paths(G, source=combination[0], target=combination[1])
            )
            paths.extend(shortest_paths)
        if len(paths) == 0:
            print("No paths exist between the concepts")
        else:
            print("Paths exist between the concepts")
            print(paths)
    return paths


def evaluate_cot(QnA_dict, metric):
    """
    QnA_dict: dict - dictionary with keys as question and values as a dictionary with keys as "generated_cot", "cot_gt", "opa", "opb", "opc", "opd"
    metric: str - can be "bleu", "rouge", "sentencebert", "kgbased"
    """
    if metric == "bleu":
        bleu_score = evaluate.load("bleu")
        generated_cots = [QnA_dict[question]["generated_cot"] for question in QnA_dict]
        gt_cots = [
            (
                [QnA_dict[question]["cot_gt"]]
                if QnA_dict[question]["cot_gt"] != None
                else [""]
            )
            for question in QnA_dict
        ]
        return bleu_score.compute(predictions=generated_cots, references=gt_cots)

    elif metric == "rouge":
        rouge_score = evaluate.load("rouge")
        generated_cots = [QnA_dict[question]["generated_cot"] for question in QnA_dict]
        gt_cots = [
            (
                [QnA_dict[question]["cot_gt"]]
                if QnA_dict[question]["cot_gt"] != None
                else [""]
            )
            for question in QnA_dict
        ]
        return rouge_score.compute(predictions=generated_cots, references=gt_cots)

    elif metric == "sentencebert":
        topk = 1
        generated_cots = [QnA_dict[question]["generated_cot"] for question in QnA_dict]
        gt_cots = [QnA_dict[question]["cot_gt"] for question in QnA_dict]
        # sentences_to_compare = generated_cots + gt_cots
        # print(sentences_to_compare)
        # print(len(sentences_to_compare))
        sentence_bert_model = SentenceTransformer("all-MiniLM-L6-v2")
        sentence_bert_model.max_seq_length = 8192

        # sentence_embeddings = sentence_bert_model.encode(sentences_to_compare)
        query_embeddings = sentence_bert_model.encode(generated_cots)
        gt_embeddings = sentence_bert_model.encode(gt_cots)

        num_queries = len(query_embeddings)

        # Calculate cosine similarity matrix between all query embeddings and GT embeddings
        similarities = cosine_similarity(query_embeddings, gt_embeddings)

        # Get the indices of the top-k highest similarities for each query
        top_k_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :topk]

        # Check if the index of the GT embedding matches the index of the query embedding within the top-k indices
        correct_matches = np.any(
            top_k_indices == np.arange(num_queries)[:, None], axis=1
        )

        # Calculate recall by counting the number of correct matches and dividing by the total number of queries
        recall = np.mean(correct_matches)
        avg_similarity = np.diagonal(similarities).mean()

        return recall, avg_similarity

        """cos_sim_matrix = cosine_similarity(sentence_embeddings, sentence_embeddings)
        print(cos_sim_matrix.shape)

        cos_sim_values = []
        retrieval_mat = []
        for i in range(len(generated_cots)):
            # cos_sim_values.append(cos_sim_matrix[i][i + len(generated_cots)])
            cos_sim_values.append(cos_sim_matrix[i][i + len(generated_cots)])
            recall_mat.append(cos_sim_matrix[i][len(generated_cots) :])
        cos_sim_values = np.array(cos_sim_values)
        print(cos_sim_values.shape)

        return cos_sim_values.mean()"""
    elif metric == "kgbased":
        # semmed_db = pd.read_csv("/data/ajayago/KB_embeddings/dataset/polypharmacy/PREDICATIONS_OCCURS.csv", usecols=["SUBJECT_NAME", "OBJECT_NAME", "PREDICATE", "OCCURS"])
        # semmed_db["SUBJECT_NAME"] = semmed_db["SUBJECT_NAME"].apply(lambda x: x.upper())
        # semmed_db["OBJECT_NAME"] = semmed_db["OBJECT_NAME"].apply(lambda x: x.upper())
        # G = nx.from_pandas_edgelist(semmed_db, source="SUBJECT_NAME", target="OBJECT_NAME", edge_attr="OCCURS", edge_key="PREDICATE")
        # concepts = set(nx.nodes(G))
        # clinicalBert_tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        # clinicalBert_model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        G, concepts = KG_init()
        clinicalBert_tokenizer, clinicalBert_model = BERT_init()
        cos_sim_values = []
        for question in QnA_dict.keys():
            candidate_phrases = []  # in Q and options
            candidate_phrases.extend(question.split(" "))
            candidate_phrases.extend(QnA_dict[question]["opa"].split(" "))
            candidate_phrases.extend(QnA_dict[question]["opb"].split(" "))
            candidate_phrases.extend(QnA_dict[question]["opc"].split(" "))
            candidate_phrases.extend(QnA_dict[question]["opd"].split(" "))

            # add bi grams
            candidate_phrases.extend(
                [" ".join(gram) for gram in find_ngrams(question.split(" "), 2)]
            )

            key_phrases = []
            for candidate in candidate_phrases:
                if candidate.upper() in concepts:
                    key_phrases.append(candidate.upper())
            paths = get_paths(key_phrases, G)

            clinicalbert_path_embeddings = []
            for path in paths:
                # print(path)
                inputs = clinicalBert_tokenizer(" ".join(path), return_tensors="pt")
                outputs = clinicalBert_model(**inputs)
                clinicalbert_path_embeddings.append(
                    outputs.last_hidden_state.mean(dim=1)
                )
            # print(torch.cat(clinicalbert_path_embeddings, dim=0).shape)
            # print(len(paths))
            generated_cot = QnA_dict[question]["generated_cot"]
            generated_cot_clinicalbert_embedding = clinicalBert_model(
                **clinicalBert_tokenizer(generated_cot, return_tensors="pt")
            ).last_hidden_state.mean(dim=1)

            similarity_cot_graph_paths = []
            for i in clinicalbert_path_embeddings:
                cosine_sim = torch.nn.functional.cosine_similarity(
                    generated_cot_clinicalbert_embedding, i
                )
                similarity_cot_graph_paths.append(cosine_sim.item())
            cos_sim_values.append(max(similarity_cot_graph_paths))
        return np.array(cos_sim_values).mean()
    elif (
        metric == "kgbased_v2"
    ):  # only extract key phrases from correct answer option and question
        G, concepts = KG_init()
        clinicalBert_tokenizer, clinicalBert_model = BERT_init()
        cos_sim_values = []
        for question in QnA_dict.keys():
            candidate_phrases = []  # in Q and options
            candidate_phrases.extend(question.split(" "))
            if QnA_dict[question]["cop"] == 0:
                candidate_phrases.extend(QnA_dict[question]["opa"].split(" "))
            elif QnA_dict[question]["cop"] == 1:
                candidate_phrases.extend(QnA_dict[question]["opb"].split(" "))
            elif QnA_dict[question]["cop"] == 2:
                candidate_phrases.extend(QnA_dict[question]["opc"].split(" "))
            elif QnA_dict[question]["cop"] == 3:
                candidate_phrases.extend(QnA_dict[question]["opd"].split(" "))

            # # add bi grams
            # candidate_phrases.extend([" ".join(gram) for gram in find_ngrams(question.split(" "), 2)])

            key_phrases = []
            for candidate in candidate_phrases:
                if candidate.upper() in concepts:
                    key_phrases.append(candidate.upper())
            paths = get_paths(key_phrases, G)

            clinicalbert_path_embeddings = []
            for path in paths:
                # print(path)
                inputs = clinicalBert_tokenizer(" ".join(path), return_tensors="pt")
                outputs = clinicalBert_model(**inputs)
                clinicalbert_path_embeddings.append(
                    outputs.last_hidden_state.mean(dim=1)
                )
            # print(torch.cat(clinicalbert_path_embeddings, dim=0).shape)
            # print(len(paths))
            generated_cot = QnA_dict[question]["generated_cot"]
            generated_cot_clinicalbert_embedding = clinicalBert_model(
                **clinicalBert_tokenizer(generated_cot, return_tensors="pt")
            ).last_hidden_state.mean(dim=1)

            similarity_cot_graph_paths = []
            for i in clinicalbert_path_embeddings:
                cosine_sim = torch.nn.functional.cosine_similarity(
                    generated_cot_clinicalbert_embedding, i
                )
                similarity_cot_graph_paths.append(cosine_sim.item())
            cos_sim_values.append(max(similarity_cot_graph_paths))
        return np.array(cos_sim_values).mean()
    else:
        print("Metric not implemented")
        return
