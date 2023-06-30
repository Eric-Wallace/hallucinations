import numpy as np
import openai
import json
from evaluate_em import exact_match_score
import time
from transformers import LlamaTokenizer, LlamaForCausalLM


def process_raw_data(in_json):
	questions = []
	answers = []
	ids = []
	for item in in_json['data']:
		questions.append(item['question'].strip())
		answers.append(item['answers'])
		ids.append(item['id'])
	return questions, answers, ids

def random_sampling(questions, answers, ids, num):
	"""randomly sample subset of the pairs"""
	idxs = np.random.randint(0, high=len(answers), size=num)
	questions = np.array(questions)
	answers = np.array(answers)
	if ids is not None:
		ids = np.array(ids)
		return questions[idxs], answers[idxs], ids[idxs]
	else:
		return questions[idxs], answers[idxs], None

def construct_prompt(prompt_prefix, train_sentences, train_labels, test_sentence):
	"""construct a single prompt to be fed into GPT3"""
	prompt = prompt_prefix
	for s, l in zip(train_sentences, train_labels):
		prompt += "Q: "
		if s[0] == '"' and s[-1] == '"': # some of the examples have quotes at the beginning and end
			s = s[1:-1]
		if s[-1] != "?": # add missing question mark
			if s[-1] == '.': # remove period if present
				s = s[0:-1]
			s += "?"
		prompt += s + "\n"

		prompt += "A: "
		assert len(l) == 1
		prompt += l[0] + "\n\n" # label should be size 1 list

	prompt += "Q: "
	if test_sentence[0] == '"' and test_sentence[-1] == '"': # some of the examples have quotes at the beginning and end
		test_sentence = test_sentence[1:-1]
	if test_sentence[-1] != "?": # add missing question mark
		test_sentence += "?"
	prompt += test_sentence + "\n"
	prompt += "A:"
	return prompt

lm = None
tokenizer = None
def setup_gpt2(model_name):
    # load the GPT-2 model
    global lm
    global tokenizer
    if lm is None:
        print("Setting up language model")
        'openlm-research/open_llama_3b'
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        lm = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')

def complete(prompt, l=10, temp=0):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generation_output = model.generate(input_ids=input_ids, max_new_tokens=l, temperature=temp, output_scores=True)
    import pdb; pdb.set_trace()
    print(tokenizer.decode(generation_output[0]))

def get_correctness_batch(samp_train_questions, samp_train_answers, test_sentences, test_labels, prompt_prefix):
	def chunks(lst, n):
		"""Yield successive n-sized chunks from lst."""
		for i in range(0, len(lst), n):
			yield lst[i:i + n]

	em_total = 0.0
	total_examples = 0.0

	test_sentences_chunks = list(chunks(test_sentences, 20))
	test_labels_chunks = list(chunks(test_labels, 20))
	for chunk_id, chunk in enumerate(test_sentences_chunks):
		prompts = []
		for sentence in chunk:
			curr_prompt = construct_prompt(prompt_prefix, samp_train_questions, samp_train_answers, sentence)
            print(curr_prompt)
            exit()
			prompts.append(curr_prompt)

		resp = complete(prompts, 15)
		for answer_id, answer in enumerate(resp.choices):
			total_examples += 1
			answer = answer.text
			answer = answer.split('\n')[0] # chop off anything after \n
			em = np.max([exact_match_score(answer, tl) for tl in test_labels_chunks[chunk_id][answer_id]])
			if not em:
				pass
				# error analysis
				# print(prompts[answer_id])
				# print(answer)
				# print(test_labels_chunks[chunk_id][answer_id])
				# # print(test_sentences_chunks[chunk_id][answer_id])
				# print()
			else:
				em_total += 1

	return em_total / total_examples

def main():

	# set random seed
	np.random.seed(2)

	# load data from repo
	train_json = json.load(open("open-domain-qa-data/triviaqa-unfiltered/train.json", "r"))
	test_json = json.load(open("open-domain-qa-data/triviaqa-unfiltered/dev.json", "r"))
	main_answer_json = json.load(open("open-domain-qa-data/triviaqa-unfiltered/triviaqa_id_to_main_answer.json", "r"))

	train_questions, train_answers, train_ids = process_raw_data(train_json)
	test_questions, test_answers, _ = process_raw_data(test_json)

	# Params
	prompt_prefix = "Answer the following questions.\n\n"
	num_test_examples = 100 # len(test_answers) # not the full test set
    train_size = 1

	# Start evaluation
	# subsample test to speed things up
	samp_test_questions, samp_test_answers, _ = random_sampling(test_questions, test_answers, None, num_test_examples)

    # get random training points
    samp_train_questions, samp_train_answers, samp_train_ids = random_sampling(train_questions, train_answers, train_ids, train_size)
    # take the best answer from the original labeling of the dataset
    for index, samp_train_answer in enumerate(samp_train_answers):
        samp_train_answers[index] = [main_answer_json[samp_train_ids[index]]]

    accuracy = get_correctness_batch(samp_train_questions, samp_train_answers, samp_test_questions, samp_test_answers, prompt_prefix)
    accuracy_fixed_train_size.append(accuracy)
    print("EM", accuracy)

if __name__ == '__main__':
	main()
