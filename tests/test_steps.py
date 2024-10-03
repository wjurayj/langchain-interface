""" """

from unittest import TestCase
from langchain_openai import ChatOpenAI
from langchain_interface.steps import Step
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache


class TestClaimSetDiffSteps(TestCase):
    def setUp(self):
        self._llm = ChatOpenAI(
            temperature=0,
            top_p=1,
            model="gpt-4o",
            max_tokens=None,
            verbose=True,
        )
        self._test_cases = [
            {
                "question": "Who first discovered calculus?",
                "positive": [
                    "Calculus was discovered by Isaac Newton.",
                    "Calculus was discovered by Pierre-Simon Laplace."
                ],
                "negative": [
                    "Calculus was discovered by Gottfried Wilhelm Leibniz.",
                    "Calculus was discovered by Albert Einstein.",
                    "Calculus was discovered by Galileo Galilei.",
                    "Calculus was discovered by Johannes Kepler."
                ],
            },
            {
                "question": "Which movie does the song \"Tujhe Dekha Toh Yeh Jana Sanam\" come from?",
                "positive": [
                    "The movie is Aashiqui.",
                    "The movie is Kuch Kuch Hota Hai."
                ],
                "negative": [
                    "The movie is Dilwale Dulhania Le Jayenge.",
                    "\"Khamoshi: The Musical\" is the movie name.",
                    "The movie is \"Tujhe Dekha To Yeh Jaana Sanam\" (1996).",
                    "The movie is \"Kabhi Khushi Kabhie Gham\"."
                ],
            },
            {
                "question": "Which is the best poem by William Wordsworth?",
                "positive": [
                    "The best poem by William Wordsworth is \"Daffodils\".",
                    "The best poem by William Wordsworth is \"The Prelude\".",
                    "The best poem by William Wordsworth is \"Lines Composed a Few Miles Above Tintern Abbey\"."
                ],
                "negative": [
                    "The best poem by William Wordsworth is \"Ode: Intimations of Immortality\".",
                    "The best poem by William Wordsworth is \"The Excursion\"."
                ]
            },
            {
                "question": "Who's the best football player of all time?",
                "positive": [
                    "The best football player of all time is Pelé.",
                    "The best football player of all time is Diego Maradona.",
                    "The best football player of all time is Lionel Messi."
                ],
                "negative": [
                    "The best football player of all time is Cristiano Ronaldo.",
                    "The best football player of all time is Zinedine Zidane.",
                    "The best football player of all time is Johan Cruyff."
                ]
            },
            {
                "question": "Who invented pocket watch?",
                "positive": [
                    "The pocket watch was invented by Peter Henlein.",
                    "The pocket watch was invented by Abraham-Louis Breguet."
                ],
                "negative": [
                    "The pocket watch was invented by John Harrison.",
                    "The pocket watch was invented by George Graham.",
                    "The pocket watch was invented by Thomas Mudge.",
                    "The pocket watch was invented by John Arnold."
                ]
            }
        ]
        
    def test_answer_shortening(self):
        """ """
        
        answer_shortening_step: Step = Step.from_params({"type": "answer-shortening"})
        chain = answer_shortening_step.chain_llm(self._llm)
        
        for test_case in self._test_cases:
            question = test_case["question"]
            positive_list = [{"question": question, "answer": answer} for answer in test_case["positive"]]
            negative_list = [{"question": question, "answer": answer} for answer in test_case["negative"]]
            all_list = positive_list + negative_list
            
            responses = chain.batch(all_list)
            print('-' * 50)
            for response, input_ in zip(responses, all_list):
                print(f"{input_['answer']} -> {response.short_answer}")
            print('-' * 50)
            
    def test_explain_diff(self):
        """ """
        
        answer_shortening_step: Step = Step.from_params({"type": "answer-shortening"})
        chain = answer_shortening_step.chain_llm(self._llm)
        
        prepared = []
        
        for test_case in self._test_cases:
            question = test_case["question"]
            # positive_list = [{"group_a": answer, "group_b": answer} for answer in test_case["positive"]]
            # negative_list = [{"group_a": answer, "group_b": answer} for answer in test_case["negative"]]
            
            positive_list = [{"question": question, "answer": answer} for answer in test_case["positive"]]
            negative_list = [{"question": question, "answer": answer} for answer in test_case["negative"]]
            all_list = positive_list + negative_list
            
            responses = chain.batch(all_list)

            pos_responses = responses[:len(positive_list)]
            neg_responses = responses[len(positive_list):]

            pos_responses = [response.short_answer for response in pos_responses]
            neg_responses = [response.short_answer for response in neg_responses]
            
            group_a_text = '\n'.join([f"- {r}" for r in pos_responses])
            group_b_text = '\n'.join([f"- {r}" for r in neg_responses])
            
            prepared.append({
                "group_a": group_a_text,
                "group_b": group_b_text
            })
            
        explain_diff_step: Step = Step.from_params({"type": "explain-diff"})
        chain = explain_diff_step.chain_llm(self._llm)

        explanation_responses = chain.batch(prepared)
        
        for response, input_, test_case in zip(explanation_responses, prepared, self._test_cases):
            print('-' * 50)
            print(f"Question: {test_case['question']}")
            print(f"Goup A: \n{input_['group_a']}\n")
            print(f"Goup B: \n{input_['group_b']}\n")
            print("Explanation:")
            print(response.messages)
            print('-' * 50)
            
    def test_vague_answer_generation(self):
        """ """
        answer_shortening_step: Step = Step.from_params({"type": "answer-shortening"})
        chain = answer_shortening_step.chain_llm(self._llm)
        
        prepared = []
        
        for test_case in self._test_cases:
            question = test_case["question"]
            # positive_list = [{"group_a": answer, "group_b": answer} for answer in test_case["positive"]]
            # negative_list = [{"group_a": answer, "group_b": answer} for answer in test_case["negative"]]
            
            positive_list = [{"question": question, "answer": answer} for answer in test_case["positive"]]
            negative_list = [{"question": question, "answer": answer} for answer in test_case["negative"]]
            all_list = positive_list + negative_list
            
            responses = chain.batch(all_list)

            pos_responses = responses[:len(positive_list)]
            neg_responses = responses[len(positive_list):]

            pos_responses = [response.short_answer for response in pos_responses]
            neg_responses = [response.short_answer for response in neg_responses]
            
            group_a_text = '\n'.join([f"- {r}" for r in pos_responses])
            group_b_text = '\n'.join([f"- {r}" for r in neg_responses])
            
            prepared.append({
                "group_a": group_a_text,
                "group_b": group_b_text
            })
            
        explain_diff_step: Step = Step.from_params({"type": "explain-diff"})
        chain = explain_diff_step.chain_llm(self._llm)

        explanation_responses = chain.batch(prepared)
        
        vague_answer_step: Step = Step.from_params({"type": "vague-answer"})
        chain = vague_answer_step.chain_llm(self._llm)

        vg_inputs = [
            {"discussion": ersp.messages, "question": test_case["question"]}
            for ersp, test_case in zip(explanation_responses, self._test_cases)
        ]

        vague_responses = chain.batch(vg_inputs)
        
        for response, input_, test_case in zip(vague_responses, vg_inputs, self._test_cases):
            print('-' * 50)
            print(f"Question: {test_case['question']}")
            print(f"Discussion: {input_['discussion']}")
            print("Vague Answer:")
            print(response.general_answer)
            print('-' * 50)
            
            
class TestAnchoredClusteringSteps(TestCase):
    def setUp(self):
        self._llm = ChatOpenAI(
            temperature=0,
            top_p=1,
            model="gpt-4o",
            max_tokens=None,
            verbose=True,
        )
        self._test_cases = [
            {
                "selected": "1972",
                "num_select": 1,
                "candidates": [
                    "1864",
                    "1882",
                    "1988",
                    "1977"
                ]
            },
            {
                "selected": "Leonel Messi",
                "num_select": 1,
                "candidates": [
                    "Cristiano Ronaldo",
                    "Diego Maradona",
                    "Pelé",
                    "Peng Shuai"
                ]
            },
            {
                "selected": "Natural Language Processing",
                "num_select": 1,
                "candidates": [
                    "Machine Learning",
                    "Artificial Intelligence",
                    "Deep Learning",
                    "Computer Vision"
                ]
            }
        ]
        
    def test_anchored_clustering(self):
        """ """
        
        anchored_clustering_step: Step = Step.from_params({"type": "anchored-clustering"})
        chain = anchored_clustering_step.chain_llm(self._llm)
        
        responses = chain.batch(self._test_cases)
        
        for response, input_ in zip(responses, self._test_cases):
            print('-' * 50)
            print(f"Selected: {input_['selected']}")
            print(f"Candidates: {', '.join(input_['candidates'])}")
            print("Response:")
            print(response.increments)
            print('-' * 50)

            
class TestQuizQuestionStep(TestCase):

    def setUp(self):
        self._llm = ChatOpenAI(
            temperature=0,
            top_p=1,
            model="gpt-4o-mini",
            max_tokens=None,
            verbose=True,
        )
        self._test_cases = [
            {
                "claim": "Hegel was a German philosopher.",
            },
            {
                "claim": "Russell developed the theory of descriptions.",
            },
        ]

    def test_quiz_question_generation(self):
        
        step: Step = Step.from_params({"type": "quiz-question"})
        runnable_chain = step.chain_llm(self._llm)
        
        response = runnable_chain.batch(self._test_cases)
        
        for res, tc in zip(response, self._test_cases):
            print(tc["claim"])
            print(res.question)
            print(res.answer_template)
            print('-' * 50)

            
class TestTestOnQuizStep(TestCase):
    def setUp(self):
        self._llm = ChatOpenAI(
            temperature=0.7,
            top_p=1,
            model="gpt-4o-mini",
            max_tokens=None,
            verbose=True,
        )
        self._test_cases = [
            {
                "question": "Who first discovered calculus?",
                "answer_template": "Calculus was discovered by <PLACEHOLDER>.",
            },
            {
                "question": "Who's the best member of blackpink?",
                "answer_template": "The best member of blackpink is <PLACEHOLDER>.",
            },
        ]
        
    def test_answering_with_template(self):
        
        step: Step = Step.from_params({"type": "test-on-quiz"})
        runnable_chain = step.chain_llm(self._llm)
        
        response = runnable_chain.batch(self._test_cases)
        
        for res, tc in zip(response, self._test_cases):
            print(tc["answer_template"].replace("<PLACEHOLDER>", res.infill))
            print('-' * 50)