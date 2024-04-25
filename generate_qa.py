from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import nltk
from nltk.tokenize import sent_tokenize
import random
import re

nltk.download('punkt')

def generate_options(correct_answer):
    """
    Generates three other options based on the correct answer's integer value.
    """
    numbers = [int(s) for s in re.findall(r'\b\d+\b', correct_answer)]
    if not numbers:
        return []
    
    main_number = numbers[0]
    options = set([correct_answer])

    while len(options) < 4:
        variation = main_number + random.choice([-1, 1]) * random.randint(1, 10)
        new_option = re.sub(str(main_number), str(variation), correct_answer, 1)
        options.add(new_option)
    
    return list(options)

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

        options = generate_options(answer)
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