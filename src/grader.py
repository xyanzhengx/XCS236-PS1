#!/usr/bin/env python3
import unittest
import argparse
import inspect
import os
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
import torch
import torch.nn as nn
from utils import *
from thefuzz import fuzz
import pickle as pkl

import submission

from datasets.encoder import get_codec
from datasets.neurips_dataset import NIPS2015Dataset
from model import GPT2, load_weight
from utils import *

device = torch.device("cpu")
SEED = 42
NEW_SEED = 56
SAMPLED_OUTPUT_OFFSET = 17

config = parse_config()
model = model = GPT2(config)
model = load_weight(model, torch.load('gpt2-pytorch_model.bin', map_location=device))
model = model.to(device)
model.eval()

### BEGIN_HIDE ###
### END_HIDE ###

#########
# TESTS #
#########

class Test_6c(GradedTestCase):
    def setUp(self):
        self.codec = get_codec()
        self.paper_dataset = NIPS2015Dataset(data_folder='datasets')
        self.partition = 8
        self.paper_idx = 8
        self.abstract_length = 50
        self.paper_iter = iter(self.paper_dataset)
        self.sol_text = "We study the problem of hierarchical clustering on data from the same dataset"
        self.sample_offset = 10
        self.threshold = 70

    @graded(timeout=15)
    def test_0(self):
        """6c-0-basic:  check sampled text expected size"""
        start_text = next(self.paper_iter)['abstract'][:self.abstract_length]
        start_text = self.codec.encode(start_text).to(device)
        text = submission.sample(model, start_text, config, length=config.n_ctx // self.partition)
        self.assertEqual(text.shape, torch.Size([1, (config.n_ctx // self.partition) + self.sample_offset]))    

    @graded(timeout=15)
    def test_1(self):
        """6c-1-basic:  check if sampled text yields similar text"""
        torch.manual_seed(SEED)
        for i in range(self.paper_idx):
            start_text = next(self.paper_iter)['abstract'][:self.abstract_length]
        start_text = self.codec.encode(start_text).to(device)
        text = submission.sample(model, start_text, config, length=config.n_ctx // self.partition)
        text = self.codec.decode(text.tolist()[0]).split(".")[0]
        text = text[:len(self.sol_text)]
        ratio = fuzz.ratio(text, self.sol_text)
        match = True if ratio >= self.threshold else False
        self.assertTrue(match, "First sentence of paper abstract does not pass similarity threshold comparison.")
    
    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_6d(GradedTestCase):
    def setUp(self):
        self.codec = get_codec()
        self.paper_dataset = NIPS2015Dataset(data_folder='datasets')
        self.partition = 8
        self.solution_tensor = torch.tensor(-47.0)
        self.threshold = 5
        self.abstract_length = 50

    @graded()
    def test_0(self):
        """6d-0-basic:  ensure log_likelihood generates expected scalars""" 
        torch.manual_seed(SEED)
        paper_iter = iter(self.paper_dataset)
        start_text = next(paper_iter)['abstract'][:self.abstract_length]
        start_text = self.codec.encode(start_text).to(device)
        logs = submission.log_likelihood(model, start_text)
        self.assertTrue(
            torch.allclose(
                torch.tensor(logs),
                self.solution_tensor,
                atol=self.threshold
            )
        ) 
    
    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_6e(GradedTestCase):
    def setUp(self):
        self.codec = get_codec()
        self.indexes = [0,35,67,143,178,202,299]
        self.results = [False, False, False, False, False, False, True]

    @graded()
    def test_0(self):
        """6e-0-basic: check if classification generates correct class for specific values of snippets.pkl"""
        with open(os.path.join('datasets', 'snippets.pkl'), 'rb') as f:
            snippets = pkl.load(f)
        classification_results = []
        for i in self.indexes:
            snippet = snippets[i]
            classResult = submission.classification(model, self.codec.encode(snippet).to(device))
            classification_results.append(classResult)
        self.assertTrue(classification_results == self.results, msg=f"incorrect classification for snippets.pkl at index: {i}")
    
    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_6f(GradedTestCase):
    def setUp(self):
        self.codec = get_codec()
        self.paper_dataset = NIPS2015Dataset(data_folder='datasets')
        self.partition = 32
        self.sol_text = "Crowdsourcing has gained immense popularity in mac and linux"
        self.abstract_length = 50
        self.temp_correct_logits_one = torch.tensor([[-2.0000e+10, -2.0000e+10, -2.0000e+10, -2.0000e+10, -2.0000e+10]])
        self.temp_correct_logits_two = torch.tensor([[-1.3333e+10, -1.3333e+10, -1.3333e+10, -1.3333e+10, -1.3333e+10]])
        self.temperature_one = 0.5
        self.temperature_two = 0.75
        self.temp_horizon = 1
        self.threshold = 0.05
        self.lev_treshold = 70

    @graded()
    def test_0(self):
        """6f-0-basic:  check temperature_scale with horizon=1 returns logits divided by temperature"""
        paper_iter = iter(self.paper_dataset)
        start_text = next(paper_iter)['abstract'][:self.abstract_length]
        start_text = self.codec.encode(start_text).to(device)
        logits, new_past = model(start_text, past=None)
        current_logits = logits[:, -1, :]
        logits = submission.top_k_logits(current_logits, k=config.top_k)
        scaled_logits_one = submission.temperature_scale(logits, model, new_past, config, self.temperature_one, self.temp_horizon)
        scaled_logits_one = scaled_logits_one[:, :5]
        self.assertTrue(
            torch.allclose(
                scaled_logits_one,
                self.temp_correct_logits_one,
                atol=self.threshold
            ),
            "temperature scaling was not applied correctly"
        )

    @graded(timeout=10)
    def test_1(self):
        """6f-1-basic:  check temperature_scale with horizon=1 yields similar samples"""
        torch.manual_seed(SEED)
        paper_iter = iter(self.paper_dataset)
        start_text = next(paper_iter)['abstract'][:self.abstract_length]
        start_text = self.codec.encode(start_text).to(device)
        text = submission.sample(model, start_text, config, length=config.n_ctx // self.partition, temperature=0.95, temperature_horizon=1)
        text = self.codec.decode(text.tolist()[0]).split('.')[0]
        ratio = fuzz.ratio(text, self.sol_text)
        match = True if ratio >= self.lev_treshold else False
        self.assertTrue(match, msg="sampled text from adding temperaturing scaling with horizon=1 does not pass similarity treshold")
    
    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_6h(GradedTestCase):
    def setUp(self):
        self.codec = get_codec()
        self.paper_dataset = NIPS2015Dataset(data_folder='datasets')
        self.length = 5
        self.sol_text = "Crowdsourcing has gained immense popularity in macs, laptop computers,"
        self.abstract_length = 50
        self.threshold = 70

    @graded(timeout=20, is_extra_credit=True)
    def test_0(self):
        """6h-0-basic:  check temperature_scale with horizon=2 yields expected samples"""
        torch.manual_seed(NEW_SEED)
        paper_iter = iter(self.paper_dataset)
        start_text = next(paper_iter)['abstract'][:self.abstract_length]
        start_text = self.codec.encode(start_text).to(device)
        text = submission.sample(model, start_text, config, length=self.length, temperature=0.95, temperature_horizon=2)
        text = self.codec.decode(text.tolist()[0]).split(".")[0]
        ratio = fuzz.ratio(text, self.sol_text)
        match = True if ratio >= self.threshold else False
        self.assertTrue(match, msg="sampled text generated after adding temperaturing scaling with horizon=2 does not pass similarity threshold")

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    
def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    print("question, part", question, part)
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)

if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)