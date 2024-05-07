import re
import json
from huth.features.questions.qa_questions_base import *
from huth.features.questions.qa_questions_data_boost import *
from huth.features.questions.qa_questions_llama_boost import *
from os.path import join, dirname
path_to_file = dirname(__file__)


def _split_bulleted_str(s, remove_parentheticals=False):
    qs = [q.strip('- ') for q in s.split('\n')]
    if remove_parentheticals:
        qs = [re.sub(r'\(.*?\)', '', q).strip() for q in qs]
    return qs


def _rewrite_to_focus_on_end(question, suffix='last'):
    '''Note, haven't screened questions beyond v4
    to make sure they're compatible with this
    '''
    if suffix == 'last':
        focus_text = 'In the last word of the input, '
    elif suffix == 'ending':
        focus_text = 'At the end of the input, '
    elif suffix == 'last10':
        focus_text = 'In the last ten words of the input, '
    else:
        raise ValueError(suffix)
    # replace nouns
    question = question.lower().replace('the sentence', 'the text').replace(
        'the story', 'the text').replace('the narrative', 'the text')
    question = question.replace(' in the input?', '?').replace(
        ' in the input text?', '?')
    return focus_text + question


def get_kwargs_list_for_version_str(version_str: str):
    # version str contains version and suffix
    # v3 -> v1, v2, v3
    # v3-ending -> v1-ending, v2-ending, v3-ending
    # v3_boost_basic -> v1, v2, v3_boost_basic
    # v3_boost_examples -> v1, v2, v3_boost_examples

    # check that there is no more than one hyphen or one underscore
    assert len(version_str.split('-')) <= 2
    assert len(version_str.split('_')) <= 2

    # deal with ending
    if '-' in version_str:
        suffix = '-' + version_str.split('-')[1]
        version_str = version_str.split('-')[0]
    else:
        suffix = ''

    if not '_' in version_str:
        version_num = int(version_str.replace('v', ''))
        kwargs_list = [{'qa_questions_version': f'v{i + 1}{suffix}'}
                       for i in range(version_num)]

    # v3_boost etc. require that we first get the base v1, v2 before adding v3_boost
    else:
        version_num = int(version_str.split('_')[0].replace('v', ''))
        boost_type = '_' + version_str.split('_')[1]
        assert version_num >= 3
        kwargs_list_base = [
            {'qa_questions_version': f'v1{suffix}'},
            {'qa_questions_version': f'v2{suffix}'}
        ]
        kwargs_list_boost = [{'qa_questions_version': f'v{i + 1}{boost_type}{suffix}'}
                             for i in range(2, version_num)]
        kwargs_list = kwargs_list_base + kwargs_list_boost

    return kwargs_list


def get_questions(version='v1', suffix=None, full=False):
    '''Different versions
    -last, -ending adds suffixes from last
    '''
    if len(version.split('-')) > 1:
        version, suffix = version.split('-')
    remove_parentheticals = False

    if version == 'v1':
        ans_list = [ANS_SEMANTIC, ANS_STORY, ANS_STORY_FOLLOWUP, ANS_WORDS]
        remove_list = []
    elif version == 'v2':
        ans_list = [ANS_NEURO, ANS_NEURO_FOLLOWUP]
        remove_list = ['v1']

    # boost versions
    elif version == 'v3_boostbasic':
        ans_list = [ANS_BOOST_LLAMA_v3_1, ANS_BOOST_LLAMA_v3_2]
        remove_list = ['v1', 'v2']

    elif version == 'v3_boostexamples':
        ans_list = [ANS_BOOST_LLAMA_v3_1_ex, ANS_BOOST_LLAMA_v3_2_ex]
        remove_parentheticals = True
        remove_list = ['v1', 'v2']

    elif version == 'v4_boostexamples':
        ans_list = [ANS_BOOST_LLAMA_v4_1_ex, ANS_BOOST_LLAMA_v4_2_ex]
        remove_parentheticals = True
        remove_list = ['v1', 'v2', 'v3_boostexamples']

    # non-boost versions
    elif version == 'v3':
        ans_list = [ANS_RANDOM_DATA_EXAMPLES, ANS_RANDOM_DATA_EXAMPLES_2]
        remove_parentheticals = True
        remove_list = ['v1', 'v2']

    elif version == 'v4':
        ans_list = [ANS_BOOST_1, ANS_BOOST_2]
        remove_list = ['v1', 'v2', 'v3']

    elif version == 'v5':
        ans_list = [ANS_BOOST_5, ANS_BOOST_5_2]
        remove_list = ['v1', 'v2', 'v3', 'v4']

    elif version == 'v6':
        ans_list = [ANS_BOOST_6, ANS_BOOST_6_2]
        remove_list = ['v1', 'v2', 'v3', 'v4', 'v5']

    # special cases
    elif version == 'base':
        return get_questions(version='v2', suffix=suffix, full=True)

    elif version == 'all':
        return get_questions(version='v6', suffix=suffix, full=True)

    qs = sum([_split_bulleted_str(ans, remove_parentheticals)
             for ans in ans_list], [])

    if suffix is not None:
        qs = [_rewrite_to_focus_on_end(q, suffix) for q in qs]

    qs_remove = sum([get_questions(version=v, suffix=suffix)
                     for v in remove_list], [])
    qs_added = sorted(list(set(qs) - set(qs_remove)))
    if full:
        # be careful to always add things in the right order!!!!
        return qs_remove + qs_added
    return qs_added


def get_question_num(question_version):
    if '-' in question_version:
        return int(question_version.split('-')[0][1:])
    else:
        return int(question_version[1])


if __name__ == "__main__":
    # for k in ['v1', 'v3', 'v6', 'v3_boostbasic', 'v3_boostexamples', 'v4-ending', 'v3_boostbasic-ending', 'v4_boostexamples']:
    # print(k, get_kwargs_list_for_version_str(k))

    for v in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']:
        print(v, len(get_questions(v)))
    print('total questions', len(get_questions('all')))

    # boosting
    for v in ['v3_boostbasic', 'v3_boostexamples', 'v4_boostexamples']:
        print(v, len(get_questions(v)))

    # write all questions to a json file
    with open(join(path_to_file, 'questions/all_questions.json'), 'w') as f:
        json.dump(get_questions('all'), f, indent=4)
    with open(join(path_to_file, 'questions/base_questions.json'), 'w') as f:
        json.dump(get_questions('base'), f, indent=4)
    with open(join(path_to_file, 'questions/v3_boostexamples.json'), 'w') as f:
        json.dump(get_questions('v3_boostexamples', full=True), f, indent=4)

    # for q in get_questions('v4_boostexamples'):
        # print(q)
