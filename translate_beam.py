import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
import math

import torch
from torch.serialization import default_restore_location

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.beam import BeamSearch, BeamSearchNode


def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', default=False, help='Use a GPU')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--data', default='/Users/wangqiaowen/atmt/data_asg4/prepared_data', help='path to data directory')
    parser.add_argument('--checkpoint-path', default='/Users/wangqiaowen/atmt/checkpoints_asg4/checkpoint_best.pt', help='path to the model file')
    parser.add_argument('--batch-size', default=None, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--output', default='model_translations.txt', type=str,
                        help='path to the output file destination')
    parser.add_argument('--max-len', default=100, type=int, help='maximum length of generated sequence')

    # Add beam search arguments
    parser.add_argument('--beam-size', default=5, type=int, help='number of hypotheses expanded in beam search')
    parser.add_argument('--alpha', default=0, type=float, help='alpha value for length normalization')
    parser.add_argument('--gamma', default=0, type=float, help='gamma value for ranked hypotheses punishment')

    return parser.parse_args()


def main(args):
    """ Main translation function' """
    # Load arguments from checkpoint
    torch.manual_seed(args.seed)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args_loaded = argparse.Namespace(**{**vars(args), **vars(state_dict['args'])})
    args_loaded.data = args.data
    args = args_loaded
    utils.init_logging(args)

    # Load dictionaries
    src_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.source_lang)))
    logging.info('Loaded a source dictionary ({:s}) with {:d} words'.format(args.source_lang, len(src_dict)))
    tgt_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.target_lang)))
    logging.info('Loaded a target dictionary ({:s}) with {:d} words'.format(args.target_lang, len(tgt_dict)))

    # Load dataset
    test_dataset = Seq2SeqDataset(
        src_file=os.path.join(args.data, 'test.{:s}'.format(args.source_lang)),
        tgt_file=os.path.join(args.data, 'test.{:s}'.format(args.target_lang)),
        src_dict=src_dict, tgt_dict=tgt_dict)

    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=1, collate_fn=test_dataset.collater,
                                              batch_sampler=BatchSampler(test_dataset, 9999999,
                                                                         args.batch_size, 1, 0, shuffle=False,
                                                                         seed=args.seed))
    # Build model and criterion
    model = models.build_model(args, src_dict, tgt_dict)
    if args.cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict['model'])
    logging.info('Loaded a model from checkpoint {:s}'.format(args.checkpoint_path))
    progress_bar = tqdm(test_loader, desc='| Generation', leave=False)


    # Iterate over the test set
    all_hyps = {}

    count = 0 

    for i, sample in enumerate(progress_bar):

        # Create a beam search object or every input sentence in batch

        batch_size = sample['src_tokens'].shape[0]
        searches = [BeamSearch(args.beam_size, args.max_len - 1, tgt_dict.unk_idx) for i in range(batch_size)]

        with torch.no_grad():
            # Compute the encoder output
            encoder_out = model.encoder(sample['src_tokens'], sample['src_lengths'])
            # __QUESTION 1: What is "go_slice" used for and what do its dimensions represent?
            go_slice = \
                torch.ones(sample['src_tokens'].shape[0], 1).fill_(tgt_dict.eos_idx).type_as(sample['src_tokens'])

            if args.cuda:
                go_slice = utils.move_to_cuda(go_slice)

            # Compute the decoder output at the first time step
            decoder_out, _ = model.decoder(go_slice, encoder_out)

            # __QUESTION 2: Why do we keep one top candidate more than the beam size?
            log_probs, next_candidates = torch.topk(torch.log(torch.softmax(decoder_out, dim=2)),
                                                    args.beam_size+1, dim=-1)

        # Create number of beam_size beam search nodes for every input sentence
        for i in range(batch_size):
            for j in range(args.beam_size):
                best_candidate = next_candidates[i, :, j]
                backoff_candidate = next_candidates[i, :, j+1]
                best_log_p = log_probs[i, :, j]
                backoff_log_p = log_probs[i, :, j+1]

                # For task 3 length normalization
                # To calculate the score after length normalization
                lp = (math.pow( (5 + log_probs.shape[1]), args.alpha ))/math.pow( (5+1), args.alpha)

                next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)

                log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
                log_p = log_p[-1]

                # Store the encoder_out information for the current input sentence and beam
                emb = encoder_out['src_embeddings'][:,i,:]
                lstm_out = encoder_out['src_out'][0][:,i,:]
                final_hidden = encoder_out['src_out'][1][:,i,:]
                final_cell = encoder_out['src_out'][2][:,i,:]
                try:
                    mask = encoder_out['src_mask'][i,:]
                except TypeError:
                    mask = None

                node = BeamSearchNode(searches[i], emb, lstm_out, final_hidden, final_cell,
                                      mask, torch.cat((go_slice[i], next_word)), log_p, 1)

                # __QUESTION 3: Why do we add the node with a negative score?

                # For task 3 and task 4 diversity promoting beam search 
                # When alpha set to 0 and gamma set to 0, the is the original code
                # When alpha set to non-zero and gamma set to 0, this is for task 3
                # When alpha set to 0 or non-zero and gamma non-zero, this is for task 4
                searches[i].add(-(node.eval()/lp-(j+1)*args.gamma), node)

        # Start generating further tokens until max sentence length reached
        for _ in range(args.max_len-1):

            # Get the current nodes to expand
            nodes = [n[1] for s in searches for n in s.get_current_beams()]

            if nodes == []:
                break # All beams ended in EOS

            # Reconstruct prev_words, encoder_out from current beam search nodes
            prev_words = torch.stack([node.sequence for node in nodes])

            encoder_out["src_embeddings"] = torch.stack([node.emb for node in nodes], dim=1)
            lstm_out = torch.stack([node.lstm_out for node in nodes], dim=1)
            final_hidden = torch.stack([node.final_hidden for node in nodes], dim=1)
            final_cell = torch.stack([node.final_cell for node in nodes], dim=1)
            encoder_out["src_out"] = (lstm_out, final_hidden, final_cell)
            try:
                encoder_out["src_mask"] = torch.stack([node.mask for node in nodes], dim=0)
            except TypeError:
                encoder_out["src_mask"] = None

            with torch.no_grad():
                # Compute the decoder output by feeding it the decoded sentence prefix
                decoder_out, _ = model.decoder(prev_words, encoder_out)

            # see __QUESTION 2
            log_probs, next_candidates = torch.topk(torch.log(torch.softmax(decoder_out, dim=2)), args.beam_size+1, dim=-1)

            for i in range(log_probs.shape[0]):
                for j in range(args.beam_size):

                    best_candidate = next_candidates[i, :, j]
                    backoff_candidate = next_candidates[i, :, j+1]
                    best_log_p = log_probs[i, :, j]
                    backoff_log_p = log_probs[i, :, j+1]

                    # For task 3 length normalization
                    # To calculate the score after length normalization
                    lp = (math.pow( (5 + log_probs.shape[1]), args.alpha ))/math.pow( (5+1), args.alpha)

                    next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)

                    log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
                    log_p = log_p[-1]

                    next_word = torch.cat((prev_words[i][1:], next_word[-1:]))


                    # Get parent node and beam search object for corresponding sentence
                    node = nodes[i]
                    search = node.search

                    # __QUESTION 4: How are "add" and "add_final" different? What would happen if we did not make this distinction?

                    # Store the node as final if EOS is generated
                    if next_word[-1 ] == tgt_dict.eos_idx:
                        node = BeamSearchNode(search, node.emb, node.lstm_out, node.final_hidden,
                                              node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                              next_word)), node.logp, node.length)

                        # For task 4 diversity promoting beam search.
                        # Gamma is the weight to control the influences of rank on the score.
                        # (j+1) is the rank for the current candidate.
                        search.add_final(-(node.eval()/lp-(j+1)*args.gamma), node)

                    # Add the node to current nodes for next iteration
                    else:

                        node = BeamSearchNode(search, node.emb, node.lstm_out, node.final_hidden,
                                              node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                              next_word)), node.logp + log_p, node.length + 1)

                        # For task 4 diversity promoting beam search.
                        # Gamma is the weight to control the influences of rank on the score.
                        # (j+1) is the rank for the current candidate.
                        search.add(-(node.eval()/lp-(j+1)*args.gamma), node)

                # print ("loop")


            # __QUESTION 5: What happens internally when we prune our beams?
            # How do we know we always maintain the best sequences?
            for search in searches:
                search.prune()

        # Segment into sentences

        best_sents = torch.stack([search.get_best()[1].sequence[1:].cpu() for search in searches])
        decoded_batch = best_sents.numpy()

        # From line 239 to line 244, the code is for task 4 diversity promoting beam search.
        # To get the n-best lists
        # top_n_sent = []
        # for search in searches :
        #     top_n = search.get_top_n(args.beam_size)
        #     for i in range(args.beam_size) :
        #         top_n_sent.append(top_n[i][1].sequence[1:])
        # best_top_sents = torch.stack(top_n_sent)

        # Line 248, the code is for task 4 diversity promoting beam search.
        # To get the n-best lists
        # decoded_batch = best_top_sents.numpy()

        output_sentences = [decoded_batch[row, :] for row in range(decoded_batch.shape[0])]

        # __QUESTION 6: What is the purpose of this for loop?
        temp = list()
        for sent in output_sentences:
            first_eos = np.where(sent == tgt_dict.eos_idx)[0]
            if len(first_eos) > 0:
                temp.append(sent[:first_eos[0]])
            else:
                temp.append(sent)
        output_sentences = temp

        # Convert arrays of indices into strings of words
        output_sentences = [tgt_dict.string(sent) for sent in output_sentences]

        for ii, sent in enumerate(output_sentences):
            all_hyps[int(sample['id'].data[ii])] = sent

        # From line 270 to line 272, the code is for task 4 diversity promoting beam search.
        # To get the n-best lists
        # for sent in enumerate(output_sentences):
        #     all_hyps[int(count)] = sent
        #     count = count+1




    # Write to file
    if args.output is not None:
        with open(args.output, 'w') as out_file:
            for sent_id in range(len(all_hyps.keys())):

                out_file.write(all_hyps[sent_id] + '\n')

                # Line 286, the code is for task 4 diversity promoting beam search.
                # To output the n-best lists
                # out_file.write(all_hyps[sent_id][1] + '\n')


if __name__ == '__main__':
    args = get_args()
    main(args)
