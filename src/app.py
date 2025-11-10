from ctypes.util import test
import sys
from pathlib import Path
import time
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import extract_markdown_from_image, ask_question, evaluate_answer
import logging 
from src.test_data import happy_case, dense_case, structural_case   
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

#allowed_imgs = {"d3.jpg"}
#test_cases = [item for item in dense_case if any(x in item["image"] for x in allowed_imgs)]

test_cases = structural_case
count_true_result = 0
markdown_cache = {}
failed_cases = []

for case in test_cases:
    image_path = case["image"]
    try:
        if image_path in markdown_cache:
            markdown = markdown_cache[image_path]
        else:
            markdown = extract_markdown_from_image(case["image"])
            markdown_cache[image_path] = markdown
            time.sleep(7.2)  # To avoid rate limiting
        
        answer = ask_question(markdown, case["ques"])
        correct = evaluate_answer(answer, case["ans"])
        if correct:
            count_true_result +=1
        else:
            failed_cases.append(case["image"])

        logging.info(f"{case['image']}")
        logging.info("Extracted Markdown Table:")
        logging.info(markdown)
        logging.info(f"Q: {case['ques']}")
        logging.info(f"A: {answer}")
        logging.info(f"Expected: {case['ans']}")
        logging.info(f"Correct: {correct}\n")
    except Exception as e:
        logging.error(f"Error while processing {case['image']}: {e}\n")

logging.info(f"Total correct answers: {count_true_result} out of {len(test_cases)}")
for failed_case in failed_cases:
    logging.info(f"Failed_case: {failed_case}")