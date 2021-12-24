import os
import torch
import argparse
import code
import prettytable
import logging
import time

from termcolor import colored
from uetqa.pipelines.pyserini_transformers import PyseriniTransformersQA


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


parser = argparse.ArgumentParser()
parser.add_argument('--reader-model', type=str, default=None,
                    help="Name of the Huggingface transformer model")
parser.add_argument('--use-fast-tokenizer', action='store_true', default=True,
                    help="Whether to use fast tokenizer")
parser.add_argument('--index-path', type=str, default=None,
                    help='Path to the index used for pyserini module')
parser.add_argument('--index-lan', type=str, default=None,
                    help='language of the index (en, vi, zh...)')
parser.add_argument('--n-passages', type=int, default=10,
                    help='number of passages to be retrieved in the retrieval stage')
parser.add_argument('--no-cuda', action='store_true',
                    help="Use CPU only")
parser.add_argument('--gpu', type=int, default=-1,
                    help="Specify GPU device id to use")
parser.add_argument('--num-workers', type=int, default=None,
                    help='Number of CPUs (for retrieving, etc)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Document paragraph batching size')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')


logger.info('Initializing pipeline...')

pipeline = PyseriniTransformersQA(
    reader_model=args.reader_model,
    use_fast_tokenizer=args.use_fast_tokenizer,
    index_path=args.index_path,
    index_lan=args.index_lan,
    cuda=args.cuda,
    ranker_config=None,
    num_workers=args.num_workers,
    batch_size=args.batch_size,
)

# ------------------------------------------------------------------------------
# Drop in to interactive mode
# ------------------------------------------------------------------------------

def q(question, top_n=3, n_passges=10):
    t0 = time.time()
    predictions = pipeline.process(
        question, top_n, n_passges, return_context=True
    )
    logger.info('Processed 1 query in %.4f (s)' % (time.time() - t0))

    table = prettytable.PrettyTable(
        ['Rank', 'Answer', 'Doc', 'Answer Score', 'Doc Score']
    )
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p['span'], p['doc_id'], '%.5g' % p['span_score'], '%.5g' % p['doc_score']])

    print('Top Predictions:')
    print(table)
    print('\nContexts:')
    for p in predictions:
        text = p['context']['text']
        start = p['context']['start']
        end = p['context']['end']
        
        output = (
            text[:start] +
            colored(text[start: end], 'green', attrs=['bold']) +
            text[end:]
        )
        print('[ Doc = %s ]' % p['doc_id'])
        print(output + '\n')


banner = """
Interactive DrQA
>> q(question, top_n=1, n_passges=5)
>> usage()
"""

def usage():
    print(banner)


code.interact(banner=banner, local=locals())