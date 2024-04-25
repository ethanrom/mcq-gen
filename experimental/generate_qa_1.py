from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline, GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
import random
import re

nltk.download('punkt')

def generate_options(question, correct_answer):
    numbers = [int(s) for s in re.findall(r'\b\d+\b', correct_answer)]
    if numbers:
        main_number = numbers[0]
        options = set([correct_answer])
    
        while len(options) < 4:
            variation = main_number + random.choice([-1, 1]) * random.randint(1, 10)
            new_option = re.sub(str(main_number), str(variation), correct_answer, 1)
            options.add(new_option)
        
        return list(options)
    else:
        return generate_options_with_gpt2(question, correct_answer)

def generate_options_with_gpt2(question, correct_answer):
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
    model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
    
    prompt = f"MCQ question with 4 options. {question}. Option 1: {correct_answer}. Option 2:"
    inputs = tokenizer_gpt2.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model_gpt2.generate(inputs, max_length=100, num_return_sequences=3, temperature=0.7)
    
    options = [correct_answer]
    for output in outputs:
        text = tokenizer_gpt2.decode(output, skip_special_tokens=True)
        option = text.split("Option")[-1].strip()
        option = option.split('.')[0].strip()  # Assuming that options end with a period
        if option not in options and option:
            options.append(option)

    # Retry mechanism if less than 4 options generated
    retries = 0
    while len(options) < 4 and retries < 5:
        outputs = model_gpt2.generate(inputs, max_length=100, num_return_sequences=3, temperature=0.7)
        for output in outputs:
            text = tokenizer_gpt2.decode(output, skip_special_tokens=True)
            option = text.split("Option")[-1].strip()
            option = option.split('.')[0].strip()
            if option not in options and option:
                options.append(option)
        retries += 1

    if len(options) < 4:  # If still less than 4 options, fill with placeholder
        while len(options) < 4:
            options.append("Option placeholder")
    
    random.shuffle(options)
    return options

def generate_questions_and_answers(text, num_questions):
    tokenizer_qg = T5Tokenizer.from_pretrained('valhalla/t5-base-qg-hl')
    model_qg = T5ForConditionalGeneration.from_pretrained('valhalla/t5-base-qg-hl')

    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    sentences = sent_tokenize(text)
    questions_and_answers = []

    for sentence in sentences:
        input_text = f"generate question: {sentence}"
        input_ids = tokenizer_qg.encode(input_text, return_tensors='pt')
        outputs = model_qg.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)

        question = tokenizer_qg.decode(outputs[0], skip_special_tokens=True)

        QA_input = {'question': question, 'context': text}
        result = nlp(QA_input)
        answer = result['answer']
        score = result['score']

        options = generate_options(question, answer)
        if options:
            random.shuffle(options)
            correct_option_index = options.index(answer) 
            correct_option_label = chr(65 + correct_option_index) 

            questions_and_answers.append({
                "question": question,
                "options": {f"Option {chr(65+i)}": option for i, option in enumerate(options)},
                "answer": f"Option {correct_option_label}",
                "confidence": score
            })
        else:
            questions_and_answers.append({
                "question": question,
                "options": [],
                "confidence": score,
                "answer": answer
            })

        if len(questions_and_answers) == num_questions:
            break

    return questions_and_answers[:num_questions]