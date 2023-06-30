import numpy as np
import openai
import json
import time
from evaluate_em import exact_match_score

def process_raw_data(in_json):
    questions = []
    answers = []
    for item in in_json['data']:
        questions.append(item['question'].strip())
        answers.append(item['answers'])
    return questions, answers

def random_sampling(questions, answers, num):
    """randomly sample subset of the training pair"""
    idxs = np.random.randint(0, high=len(answers), size=num)
    questions = np.array(questions)
    answers = np.array(answers)
    return questions[idxs], answers[idxs]

def construct_prompt(prompt_prefix, train_sentences, train_labels, test_sentence):
    """construct a single prompt to be fed into GPT3"""
    prompt = prompt_prefix
    for s, l in zip(train_sentences, train_labels):
        question, context = s[0], s[1]
        prompt +=  "Q: " + question + "?\nA:" + ' ' + l + '\n\n'

    question, context = test_sentence[0], test_sentence[1]        
    prompt +=  "Q: " + question + "?\nA:"
    print(prompt)    
    exit()
    return prompt

def complete(prompt, l=10, temp=0):
    """ocmplete the prompt using GPT3"""
    openai.api_key = "sk-JK7k3XqwFYUQGW1rLXOM1i9vSFYzQ35EMycasxbY"
    response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=l, temperature=temp)
    return response

def get_correctness_batch(samp_train_questions, samp_train_answers, test_sentences, test_labels, prompt_prefix):
    """get whether GPT3 is correct from a list of training pairs and a test pair."""
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    em_total = 0.0
    total_examples = 0.0

    test_sentences_chunks = list(chunks(test_sentences, 20))
    test_labels_chunks = list(chunks(test_labels, 20))
    for chunk_id, chunk in enumerate(test_sentences_chunks):
        print('Progress', chunk_id, len(test_sentences_chunks))
        prompts = []
        for sentence in chunk:
            # print(construct_prompt(prompt_prefix, samp_train_questions, samp_train_answers, sentence))            
            prompts.append(construct_prompt(prompt_prefix, samp_train_questions, samp_train_answers, sentence))

        resp = complete(prompts, 15)
        for answer_id, answer in enumerate(resp.choices):
            total_examples += 1
            answer = answer.text
            answer = answer.split('\n')[0] # chop off anything after newline
            em = np.max([exact_match_score(answer, tl) for tl in test_labels_chunks[chunk_id][answer_id]])
            if not em:                
                pass  
                # print(prompts[answer_id])
                # print(answer)
                # print(test_labels_chunks[chunk_id][answer_id])
                # print(test_sentences_chunks[chunk_id][answer_id])
                # print()
              
            else:
                em_total += 1
        print('Current Accuracy', em_total / total_examples)
    return em_total / total_examples

def load_data(split):
    train_json = json.load(open('open-domain-qa-data/nq-open/'+"/{}.json".format(split), "r"))
    train_questions, train_answers = process_raw_data(train_json)
    train_samples = {int(k):v for k, v in json.load(open('open-domain-qa-data/nq-open/' + "{}_samples.json".format(split), "r")).items()}
    train_ids = sorted(train_samples.keys())
    train_questions = [(train_questions[i], train_samples[i]) for i in train_ids]
    train_answers = [train_answers[i] for i in train_ids]
    return train_questions, train_answers

def main():
    # set random seed
    np.random.seed(0)

    # load data from repo
    # train_json = json.load(open("open-domain-qa-data/nq-open/train.json", "r"))
    # test_json = json.load(open("open-domain-qa-data/nq-open/test.json", "r")) 
    # train_questions, train_answers = 
