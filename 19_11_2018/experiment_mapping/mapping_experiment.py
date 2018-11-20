'''
   Learning curve experiment
   Name:         mapping_experiment.py
   Author:       Rodrigo Azevedo
   Updated:      July 23, 2018
   License:      GPLv3
'''

import os
import sys
import time

from datasets.get_datasets import *
from revision import *
from transfer import *
from mapping import *
from boostsrl import boostsrl
import numpy as np
import random
import json

source_balanced = 1
balanced = 1
firstRun = False
n_runs = 25
folds = 3

nodeSize = 2
numOfClauses = 8
maxTreeDepth = 3

if not os.path.exists('log'):
    os.makedirs('log')

#def setup_logger(name, log_file, level=logging.DEBUG):
#    formatter = logging.Formatter('%(message)s')
#    
#    handler = logging.FileHandler(log_file)        
#    handler.setFormatter(formatter)
#
#    logger = logging.getLogger(name)
#    logger.setLevel(level)
#    logger.addHandler(handler)
#
#    return logger
    
def print_function(message):
    global experiment_title
    with open('log/' + experiment_title + '.txt', 'a') as f:
        print(message, file=f)
        print(message)

experiments = [           
            #{'id': '1', 'source':'imdb', 'target':'uwcse', 'predicate':'workedunder', 'to_predicate':'advisedby'},
            #{'id': '2', 'source':'uwcse', 'target':'imdb', 'predicate':'advisedby', 'to_predicate':'workedunder'},
            #{'id': '3', 'source':'yeast', 'target':'twitter', 'predicate':'interaction', 'to_predicate':'follows'},
            #{'id': '4', 'source':'twitter', 'target':'yeast', 'predicate':'follows', 'to_predicate':'interaction'},
            #{'id': '5', 'source':'imdb', 'target':'cora', 'predicate':'workedunder', 'to_predicate':'samevenue'},
            {'id': '6', 'source':'uwcse', 'target':'cora', 'predicate':'advisedby', 'to_predicate':'samevenue'},
            #{'id': '7', 'source':'imdb', 'target':'cora', 'predicate':'workedunder', 'to_predicate':'sameauthor'},
            ]
            
bk = {
      'imdb': ['workedunder(+person,+person).',
              'workedunder(+person,-person).',
              'workedunder(-person,+person).',
              'female(+person).',
              'actor(+person).',
              'director(+person).',
              'movie(+movie,+person).',
              'movie(+movie,-person).',
              'movie(-movie,+person).',
              'genre(+person,+genre).'],
      'uwcse': ['professor(+person).',
        'student(+person).',
        'advisedby(+person,+person).',
        'advisedby(+person,-person).',
        'advisedby(-person,+person).',
        'tempadvisedby(+person,+person).',
        'tempadvisedby(+person,-person).',
        'tempadvisedby(-person,+person).',
        'ta(+course,+person,+quarter).',
        'ta(-course,-person,+quarter).',
        'ta(+course,-person,-quarter).',
        'ta(-course,+person,-quarter).',
        'hasposition(+person,+faculty).',
        'hasposition(+person,-faculty).',
        'hasposition(-person,+faculty).',
        'publication(+title,+person).',
        'publication(+title,-person).',
        'publication(-title,+person).',
        'inphase(+person,+prequals).',
        'inphase(+person,-prequals).',
        'inphase(-person,+prequals).',
        'courselevel(+course,+level).',
        'courselevel(+course,-level).',
        'courselevel(-course,+level).',
        'yearsinprogram(+person,+year).',
        'yearsinprogram(-person,+year).',
        'yearsinprogram(+person,-year).',
        'projectmember(+project,+person).',
        'projectmember(+project,-person).',
        'projectmember(-project,+person).',
        'sameproject(+project,+project).',
        'sameproject(+project,-project).',
        'sameproject(-project,+project).',
        'samecourse(+course,+course).',
        'samecourse(+course,-course).',
        'samecourse(-course,+course).',
        'sameperson(+person,+person).',
        'sameperson(+person,-person).',
        'sameperson(-person,+person).',],
      'cora': ['sameauthor(+author,+author).',
              'sameauthor(+author,-author).',
              'sameauthor(-author,+author).',
              'samebib(+class,+class).',
              'samebib(+class,-class).',
              'samebib(-class,+class).',
              'sametitle(+title,+title).',
              'sametitle(+title,-title).',
              'sametitle(-title,+title).',
              'samevenue(+venue,+venue).',
              'samevenue(+venue,-venue).',
              'samevenue(-venue,+venue).',
              'author(+class,+author).',
              'author(+class,-author).',
              'author(-class,+author).',
              'title(+class,+title).',
              'title(+class,-title).',
              'title(-class,+title).',
              'venue(+class,+venue).',
              'venue(+class,-venue).',
              'venue(-class,+venue).',
              'haswordauthor(+author,+word).',
              'haswordauthor(+author,-word).',
              'haswordauthor(-author,+word).',
              'haswordtitle(+title,+word).',
              'haswordtitle(+title,-word).',
              'haswordtitle(-title,+word).',
              'haswordvenue(+venue,+word).',
              'haswordvenue(+venue,-word).',
              'haswordvenue(-venue,+word).'],
      'twitter': ['accounttype(+account,+type).',
                  'accounttype(+account,-type).',
                  'accounttype(-account,+type).',
                  'typeaccount(+type,`account).',
                  'typeaccount(`type,+account).',
                  'tweets(+account,+word).',
                  'tweets(+account,-word).',
                  'tweets(-account,+word).',
                  'follows(+account,+account).',
                  'follows(+account,-account).',
                  'follows(-account,+account).'],
      'yeast': ['location(+protein,+loc).',
                'location(+protein,-loc).',
                'location(-protein,+loc).',
                'interaction(+protein,+protein).',
                'interaction(+protein,-protein).',
                'interaction(-protein,+protein).',
                'proteinclass(+protein,+class).',
                'proteinclass(+protein,-class).',
                'proteinclass(-protein,+class).',
                'classprotein(+class,`protein).',
                'classprotein(`class,+protein).',
                'enzyme(+protein,+enz).',
                'enzyme(+protein,-enz).',
                'enzyme(-protein,+enz).',
                'function(+protein,+fun).',
                'function(+protein,-fun).',
                'function(-protein,+fun).',
                'complex(+protein,+com).',
                'complex(+protein,-com).',
                'complex(-protein,+com).',
                'phenotype(+protein,+phe).',
                'phenotype(+protein,-phe).',
                'phenotype(-protein,+phe).'],
      }

if os.path.isfile('mapping_experiment.json'):
    with open('mapping_experiment.json', 'r') as fp:
        results = json.load(fp)
else:
    results = { 'results': {}, 'save': { }}
    firstRun = True

def save(data):
    with open('mapping_experiment.json', 'w') as fp:
        json.dump(data, fp)
        
if firstRun:
    results['save'] = {'experiment': 0, 'n_runs': 0, 'seed': random.randint(111111,999999) }
    
    
def get_all_mappings(sPreds, tPreds, srcFacts, tarFacts, forceHead=None):
    srcPreds = sPreds
    tarPreds = mapping.clean_preds(tPreds)
    fHead = None if not forceHead else mapping.find_pred(forceHead, tarPreds)
    possible_mappings = mapping.mapping(srcPreds, tarPreds, forceHead=fHead)
    # return None if incompatible forceHead is defined
    if not len(possible_mappings):
        return ({}, None)
    unaries = []
    for srcPred in srcPreds:
        s = mapping.get_types(srcPred)
        if len(s[1]) == 1:
            unaries.append(s[0])
    ret = []
    for el in possible_mappings:
        best_mapping = el
        mapd = []
        for key, value in best_mapping.items():
            if key in unaries:
                string = key + '(A) -> ' + value + '(A)'
            else:
                string = key + '(A,B) -> ' + (value if value[0] != '_' else value[1:]) + ('(A,B)' if value[0] != '_' else '(B,A)')
            mapd.append(string)
        ret.append(mapd)
    return ret

start = time.time()
#while results['save']['experiment'] < len(experiments):
while results['save']['n_runs'] < n_runs:
    print('Run: ' + str(results['save']['n_runs']))
    experiment = results['save']['experiment'] % len(experiments)
    try:
        #experiment = results['save']['experiment']
        experiment_title = experiments[experiment]['id'] + '_' + experiments[experiment]['source'] + '_' + experiments[experiment]['target']
        if experiment_title not in results['results']:
            results['results'][experiment_title] = []
    
        source = experiments[experiment]['source']
        target = experiments[experiment]['target']
        predicate = experiments[experiment]['predicate']
        to_predicate = experiments[experiment]['to_predicate']
        
        # Load source dataset
        src_total_data = datasets.load(source, bk[source], seed=results['save']['seed'])
        src_data = datasets.load(source, bk[source], target=predicate, balanced=source_balanced, seed=results['save']['seed'])
            
        # Group and shuffle
        src_facts = datasets.group_folds(src_data[0])
        src_pos = datasets.group_folds(src_data[1])
        src_neg = datasets.group_folds(src_data[2])

        # learning from source dataset
        background = boostsrl.modes(bk[source], [predicate], useStdLogicVariables=False, maxTreeDepth=maxTreeDepth, nodeSize=nodeSize, numOfClauses=numOfClauses)
        [model, total_revision_time, source_structured, will, variances] = revision.learn_model(background, boostsrl, predicate, src_pos, src_neg, src_facts, refine=None, trees=10, print_function=None)
        
        ob_save = {}
        preds = mapping.get_preds(source_structured, bk[source])
        ob_save['Predicates'] = preds
        
        # Load total target dataset
        tar_total_data = datasets.load(target, bk[target], seed=results['save']['seed'])
        
        if target in ['nell_sports', 'nell_finances', 'yago2s']:
            n_folds = folds
        else:
            n_folds = len(tar_total_data[0])
    
        results_save = []
        
        i = random.randint(0, n_folds-1)        
        
        if target not in ['nell_sports', 'nell_finances', 'yago2s']:
            [tar_train_pos, tar_test_pos] = datasets.get_kfold_small(i, tar_total_data[0])
        else:
            t_total_data = datasets.load(target, bk[target], target=to_predicate, balanced=balanced, seed=results['save']['seed'])
            tar_train_pos = datasets.split_into_folds(t_total_data[1][0], n_folds=n_folds, seed=results['save']['seed'])[i] + t_total_data[0][0]
        
        # transfer
        mapping_rules, mapping_results = mapping.get_best(preds, bk[target], datasets.group_folds(src_total_data[0]), tar_train_pos, forceHead=to_predicate)
        ob_save['Mapping results'] = mapping_results
        ob_save['Mapping rules'] = mapping_rules
        
        print_function(mapping_results)
        print_function(mapping_rules)
        
        new_target = to_predicate
        
        # Load new predicate target dataset
        tar_data = datasets.load(target, bk[target], target=new_target, balanced=balanced, seed=results['save']['seed'])
        
        # Group and shuffle
        if target not in ['nell_sports', 'nell_finances', 'yago2s']:
            [tar_train_facts, tar_test_facts] =  datasets.get_kfold_small(i, tar_data[0])
            [tar_train_pos, tar_test_pos] =  datasets.get_kfold_small(i, tar_data[1])
            [tar_train_neg, tar_test_neg] =  datasets.get_kfold_small(i, tar_data[2])
        else:
            [tar_train_facts, tar_test_facts] =  [tar_data[0][0], tar_data[0][0]]
            to_folds_pos = datasets.split_into_folds(tar_data[1][0], n_folds=n_folds, seed=results['save']['seed'])
            to_folds_neg = datasets.split_into_folds(tar_data[2][0], n_folds=n_folds, seed=results['save']['seed'])
            [tar_train_pos, tar_test_pos] =  datasets.get_kfold_small(i, to_folds_pos)
            [tar_train_neg, tar_test_neg] =  datasets.get_kfold_small(i, to_folds_neg)

        all_mapping_rules = get_all_mappings(preds, bk[target], datasets.group_folds(src_total_data[0]), tar_train_pos, forceHead=to_predicate)
        
        ob_save['Evaluating all'] = {}
        ob_save['Evaluating all']['Time spent'] = mapping_results['Generating mappings time']
        ob_save['Evaluating all']['Best AUC ROC'] = -99999
        ob_save['Evaluating all']['Best AUC PR'] = -99999
        ob_save['Evaluating all']['Best CLL'] = -99999
        
        background = boostsrl.modes(bk[target], [new_target], useStdLogicVariables=False, maxTreeDepth=maxTreeDepth, nodeSize=nodeSize, numOfClauses=numOfClauses)

        for rule in all_mapping_rules:
            # no need for only head predicate mapping
            if len(rule) > 1:
                transferred_structured = transfer.transfer(source_structured, rule)
                # learning from scratch
                [model, t_results, structured, will, variances] = revision.learn_test_model(background, boostsrl, new_target, tar_train_pos, tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, refine=revision.get_boosted_refine_file(transferred_structured), trees=10, print_function=None)
                ob_save['Evaluating all']['Best AUC ROC'] = max(ob_save['Evaluating all']['Best AUC ROC'], t_results['AUC ROC'])
                ob_save['Evaluating all']['Best AUC PR'] = max(ob_save['Evaluating all']['Best AUC PR'], t_results['AUC PR'])
                ob_save['Evaluating all']['Best CLL'] = max(ob_save['Evaluating all']['Best CLL'], t_results['CLL'])
                ob_save['Evaluating all']['Time spent'] += t_results['Learning time'] + t_results['Inference time']
                print_function('%s, AUC ROC: %s, AUC PR: %s, CLL: %s' % (rule, t_results['AUC ROC'], t_results['AUC PR'], t_results['CLL']))
                if mapping_rules == rule:
                    ob_save['Evaluating all']['Found AUC ROC'] = t_results['AUC ROC']
                    ob_save['Evaluating all']['Found AUC PR'] = t_results['AUC PR']
                    ob_save['Evaluating all']['Found CLL'] = t_results['CLL']
        results['results'][experiment_title].append(ob_save)
        print_function(ob_save)
    except Exception as e:
        print_function(e)
        print_function('Error in experiment of ' + experiment_title)
        pass
    results['save']['experiment'] += 1
    results['save']['n_runs'] += 1
    save(results)