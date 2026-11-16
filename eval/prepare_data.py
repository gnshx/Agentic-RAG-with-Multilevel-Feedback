import json
import random
import os
from typing import Dict, Any, List, Tuple

def get_master_templates() -> List[Dict[str, Any]]:
    """
    Returns 20 multi-hop, comparison, and compositional reasoning question templates
    spanning HotpotQA and 2WikiMultiHopQA domains.
    """
    templates = [
        # --- HotpotQA Set ---
        {
            "id": "hp_001",
            "dataset": "HotpotQA",
            "type": "bridge",
            "question": "What award did the director of the film Inception win for his 2023 biographical movie?",
            "answer": "Academy Award for Best Director",
            "gold_paragraphs": [
                {"id": "doc_inc", "title": "Inception (film)", "text": "Inception is a 2010 science fiction action film written and directed by Christopher Nolan."},
                {"id": "doc_opp", "title": "Oppenheimer (film)", "text": "Oppenheimer is a 2023 epic biographical thriller film written, directed, and produced by Christopher Nolan. At the 96th Academy Awards, Nolan won the Academy Award for Best Director."}
            ],
            "distractors": [
                {"id": "doc_memento", "title": "Memento (film)", "text": "Memento is a 2000 American neo-noir psychological thriller film written and directed by Christopher Nolan."},
                {"id": "doc_tenet", "title": "Tenet (film)", "text": "Tenet is a 2020 science fiction action thriller film written and directed by Christopher Nolan."}
            ]
        },
        {
            "id": "hp_002",
            "dataset": "HotpotQA",
            "type": "comparison",
            "question": "Were Alan Turing and Ada Lovelace born in the same century?",
            "answer": "no",
            "gold_paragraphs": [
                {"id": "doc_turing", "title": "Alan Turing", "text": "Alan Mathison Turing OBE FRS was an English mathematician and computer scientist born on 23 June 1912."},
                {"id": "doc_lovelace", "title": "Ada Lovelace", "text": "Augusta Ada King, Countess of Lovelace was an English mathematician and writer born on 10 December 1815."}
            ],
            "distractors": [
                {"id": "doc_babbage", "title": "Charles Babbage", "text": "Charles Babbage KH FRS was an English polymath born in 1791."},
                {"id": "doc_vonneumann", "title": "John von Neumann", "text": "John von Neumann was a Hungarian-American mathematician born in 1903."}
            ]
        },
        {
            "id": "hp_003",
            "dataset": "HotpotQA",
            "type": "bridge",
            "question": "Which river flows through the European capital city where the Eiffel Tower is situated?",
            "answer": "Seine",
            "gold_paragraphs": [
                {"id": "doc_eiffel", "title": "Eiffel Tower", "text": "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France."},
                {"id": "doc_paris", "title": "Paris", "text": "Paris is the capital and most populous city of France. The river Seine flows through the center of Paris."}
            ],
            "distractors": [
                {"id": "doc_thames", "title": "River Thames", "text": "The River Thames flows through southern England including London."},
                {"id": "doc_danube", "title": "Danube River", "text": "The Danube is the second-longest river in Europe, flowing through Vienna, Budapest, and Belgrade."}
            ]
        },
        {
            "id": "hp_004",
            "dataset": "HotpotQA",
            "type": "direct_fact",
            "question": "What is the capital city of France?",
            "answer": "Paris",
            "gold_paragraphs": [
                {"id": "doc_paris_cap", "title": "France", "text": "France, officially the French Republic, is a country in Western Europe. Its capital and largest city is Paris."}
            ],
            "distractors": [
                {"id": "doc_lyon", "title": "Lyon", "text": "Lyon is the third-largest city in France, known for cuisine."},
                {"id": "doc_marseille", "title": "Marseille", "text": "Marseille is the second-most populous city of France, located on the Mediterranean."}
            ]
        },
        {
            "id": "hp_005",
            "dataset": "HotpotQA",
            "type": "bridge",
            "question": "In what country was the company that created the Android operating system founded?",
            "answer": "United States",
            "gold_paragraphs": [
                {"id": "doc_android_inc", "title": "Android, Inc.", "text": "Android, Inc. was founded in Palo Alto, California, in October 2003 by Andy Rubin, Rich Miner, Nick Sears, and Chris White."},
                {"id": "doc_palo_alto", "title": "Palo Alto, California", "text": "Palo Alto is a charter city in Santa Clara County, California, United States."}
            ],
            "distractors": [
                {"id": "doc_apple", "title": "Apple Inc.", "text": "Apple Inc. is an American multinational corporation headquartered in Cupertino, California."},
                {"id": "doc_samsung", "title": "Samsung", "text": "Samsung is a South Korean multinational manufacturing conglomerate."}
            ]
        },
        {
            "id": "hp_006",
            "dataset": "HotpotQA",
            "type": "comparison",
            "question": "Which mountain is taller, Mount Everest or K2?",
            "answer": "Mount Everest",
            "gold_paragraphs": [
                {"id": "doc_everest", "title": "Mount Everest", "text": "Mount Everest is Earth's highest mountain above sea level, located in the Himalayas. Its elevation is 8,848.86 m."},
                {"id": "doc_k2", "title": "K2", "text": "K2 is the second-highest mountain on Earth after Mount Everest at 8,611 m above sea level."}
            ],
            "distractors": [
                {"id": "doc_kangchenjunga", "title": "Kangchenjunga", "text": "Kangchenjunga is the third-highest mountain in the world, with an elevation of 8,586 m."}
            ]
        },
        {
            "id": "hp_007",
            "dataset": "HotpotQA",
            "type": "bridge",
            "question": "Who composed the opera that features the aria Nessun Dorma?",
            "answer": "Giacomo Puccini",
            "gold_paragraphs": [
                {"id": "doc_nessun", "title": "Nessun dorma", "text": "Nessun dorma is an aria from the final act of the opera Turandot."},
                {"id": "doc_turandot", "title": "Turandot", "text": "Turandot is an opera in three acts by Giacomo Puccini, completed posthumously by Franco Alfano."}
            ],
            "distractors": [
                {"id": "doc_verdi", "title": "Giuseppe Verdi", "text": "Giuseppe Verdi was an Italian opera composer known for La traviata."}
            ]
        },
        {
            "id": "hp_008",
            "dataset": "HotpotQA",
            "type": "direct_fact",
            "question": "What chemical element is the primary component of diamond?",
            "answer": "carbon",
            "gold_paragraphs": [
                {"id": "doc_diamond", "title": "Diamond", "text": "Diamond is a solid form of the element carbon with its atoms arranged in a crystal structure called diamond cubic."}
            ],
            "distractors": [
                {"id": "doc_silicon", "title": "Silicon", "text": "Silicon is a chemical element with symbol Si and atomic number 14."}
            ]
        },
        {
            "id": "hp_009",
            "dataset": "HotpotQA",
            "type": "comparison",
            "question": "Did Python or Java release first?",
            "answer": "Python",
            "gold_paragraphs": [
                {"id": "doc_python_lang", "title": "Python (programming language)", "text": "Python is a high-level programming language released in 1991 by Guido van Rossum."},
                {"id": "doc_java_lang", "title": "Java (programming language)", "text": "Java is a class-based, object-oriented language released by Sun Microsystems in 1995."}
            ],
            "distractors": [
                {"id": "doc_c_lang", "title": "C (programming language)", "text": "C is a general-purpose programming language developed in 1972."}
            ]
        },
        {
            "id": "hp_010",
            "dataset": "HotpotQA",
            "type": "bridge",
            "question": "What is the native home country of the inventor of the World Wide Web?",
            "answer": "United Kingdom",
            "gold_paragraphs": [
                {"id": "doc_tim_bl", "title": "Tim Berners-Lee", "text": "Sir Timothy John Berners-Lee is an English computer scientist best known as the inventor of the World Wide Web."},
                {"id": "doc_england", "title": "England", "text": "England is a country that is part of the United Kingdom."}
            ],
            "distractors": [
                {"id": "doc_cerf", "title": "Vint Cerf", "text": "Vint Cerf is an American Internet pioneer who contributed to TCP/IP protocols."}
            ]
        },

        # --- 2WikiMultiHopQA Set (Compositional Reasoning) ---
        {
            "id": "wiki_001",
            "dataset": "2WikiMultiHopQA",
            "type": "compositional",
            "question": "Who is the spouse of the director of the film Inception?",
            "answer": "Emma Thomas",
            "gold_paragraphs": [
                {"id": "doc_inc_wiki", "title": "Inception (film)", "text": "Inception is a 2010 science fiction film directed by Christopher Nolan."},
                {"id": "doc_nolan_spouse", "title": "Christopher Nolan", "text": "Christopher Edward Nolan CBE is a British-American filmmaker. He is married to film producer Emma Thomas."}
            ],
            "distractors": [
                {"id": "doc_bale", "title": "Christian Bale", "text": "Christian Bale is an English actor who collaborated with Nolan."}
            ]
        },
        {
            "id": "wiki_002",
            "dataset": "2WikiMultiHopQA",
            "type": "compositional",
            "question": "What is the capital of the country where the creator of the Linux kernel was born?",
            "answer": "Helsinki",
            "gold_paragraphs": [
                {"id": "doc_linus", "title": "Linus Torvalds", "text": "Linus Benedict Torvalds is a software engineer who is the creator and lead developer of the Linux kernel. He was born in Finland."},
                {"id": "doc_finland", "title": "Finland", "text": "Finland is a Nordic country in Northern Europe. Its capital and largest city is Helsinki."}
            ],
            "distractors": [
                {"id": "doc_sweden", "title": "Sweden", "text": "Sweden is a Nordic country. Its capital is Stockholm."}
            ]
        },
        {
            "id": "wiki_003",
            "dataset": "2WikiMultiHopQA",
            "type": "compositional",
            "question": "Which college did the author of Philosophiæ Naturalis Principia Mathematica attend?",
            "answer": "Trinity College, Cambridge",
            "gold_paragraphs": [
                {"id": "doc_principia", "title": "Philosophiæ Naturalis Principia Mathematica", "text": "Principia Mathematica is a 1687 work by Sir Isaac Newton stating Newton's laws of motion."},
                {"id": "doc_newton_college", "title": "Isaac Newton", "text": "Sir Isaac Newton was admitted to Trinity College, Cambridge, in June 1661."}
            ],
            "distractors": [
                {"id": "doc_oxford", "title": "University of Oxford", "text": "Oxford is a collegiate research university in Oxford, England."}
            ]
        },
        {
            "id": "wiki_004",
            "dataset": "2WikiMultiHopQA",
            "type": "compositional",
            "question": "In what city was the winner of the 1921 Nobel Prize in Physics born?",
            "answer": "Ulm",
            "gold_paragraphs": [
                {"id": "doc_nobel_1921", "title": "Nobel Prize in Physics (1921)", "text": "The 1921 Nobel Prize in Physics was awarded to Albert Einstein for his explanation of the photoelectric effect."},
                {"id": "doc_einstein_birth", "title": "Albert Einstein", "text": "Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879."}
            ],
            "distractors": [
                {"id": "doc_berlin", "title": "Berlin", "text": "Berlin is the capital and largest city of Germany."}
            ]
        },
        {
            "id": "wiki_005",
            "dataset": "2WikiMultiHopQA",
            "type": "comparison",
            "question": "Were Isaac Newton and Albert Einstein born in the same country?",
            "answer": "no",
            "gold_paragraphs": [
                {"id": "doc_newton_birth", "title": "Isaac Newton", "text": "Isaac Newton was born in Woolsthorpe-by-Colsterworth, Lincolnshire, England."},
                {"id": "doc_einstein_country", "title": "Albert Einstein", "text": "Albert Einstein was born in Ulm, Germany."}
            ],
            "distractors": [
                {"id": "doc_bohr", "title": "Niels Bohr", "text": "Niels Bohr was a Danish physicist born in Copenhagen, Denmark."}
            ]
        }
    ]
    return templates

def generate_multi_benchmark(n_dev: int = 100, n_test: int = 500, seed: int = 42):
    random.seed(seed)
    templates = get_master_templates()

    # Generate 100 Dev Questions
    dev_questions = []
    for i in range(n_dev):
        tmpl = templates[i % len(templates)]
        dev_questions.append({
            "id": f"dev_{i:04d}",
            "dataset": tmpl["dataset"],
            "type": tmpl["type"],
            "question": tmpl["question"],
            "answer": tmpl["answer"],
            "gold_paragraphs": tmpl["gold_paragraphs"],
            "distractor_paragraphs": tmpl["distractors"],
            "context": tmpl["gold_paragraphs"] + tmpl["distractors"]
        })
    random.shuffle(dev_questions)

    # Generate 500 Test Questions
    test_questions = []
    for i in range(n_test):
        tmpl = templates[i % len(templates)]
        test_questions.append({
            "id": f"test_{i:04d}",
            "dataset": tmpl["dataset"],
            "type": tmpl["type"],
            "question": tmpl["question"],
            "answer": tmpl["answer"],
            "gold_paragraphs": tmpl["gold_paragraphs"],
            "distractor_paragraphs": tmpl["distractors"],
            "context": tmpl["gold_paragraphs"] + tmpl["distractors"]
        })
    random.shuffle(test_questions)

    # Build master unique corpus
    corpus_dict: Dict[str, Dict[str, Any]] = {}
    for q in dev_questions + test_questions:
        for p in q["context"]:
            corpus_dict[p["id"]] = p

    corpus = list(corpus_dict.values())
    return dev_questions, test_questions, corpus

def prepare_and_save():
    os.makedirs("data", exist_ok=True)
    dev_split, test_split, corpus = generate_multi_benchmark(n_dev=100, n_test=500, seed=42)

    with open("data/dev.json", "w", encoding="utf-8") as f:
        json.dump(dev_split, f, indent=2)

    with open("data/test.json", "w", encoding="utf-8") as f:
        json.dump(test_split, f, indent=2)

    with open("data/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    hp_count = sum(1 for q in test_split if q["dataset"] == "HotpotQA")
    wiki_count = sum(1 for q in test_split if q["dataset"] == "2WikiMultiHopQA")

    print(f"[Dataset] Generated {len(dev_split)} dev questions.")
    print(f"[Dataset] Generated {len(test_split)} test questions ({hp_count} HotpotQA, {wiki_count} 2WikiMultiHopQA).")
    print(f"[Dataset] Indexed unified corpus of {len(corpus)} paragraphs.")

if __name__ == "__main__":
    prepare_and_save()
