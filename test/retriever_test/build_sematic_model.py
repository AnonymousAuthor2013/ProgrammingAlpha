from programmingalpha.retrievers.semanticRanker import *
import programmingalpha
from torch.utils.data.distributed import DistributedSampler
import random
from tqdm import tqdm, trange
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import RandomSampler

def main():

    ## Required parameters
    data_dir=programmingalpha.DataPath+"inference_pair/"
    bert_model=programmingalpha.BertBasePath
    task_name="semantic"
    output_dir=programmingalpha.ModelPath

    ## Other parameters
    max_seq_length=128
    do_train=True
    do_eval=True
    do_lower_case=True
    train_batch_size=32
    eval_batch_size=8
    learning_rate=5e-5
    num_train_epochs=3
    warmup_proportion=0.1
    use_cuda=True
    cuda_rank=0
    seed=random.randint(2,118931)
    gradient_accumulation_steps=1
    fp16=False
    loss_scale=0
    loss_partial=[1,5]

    processors = {
        "semantic": SemanticPairProcessor,
    }

    num_labels_task = {
        "semantic": 4,
    }

    device=torch.device("cuda" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, fp16))

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            gradient_accumulation_steps))

    train_batch_size = int(train_batch_size /gradient_accumulation_steps)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if not do_train and not do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(output_dir) and os.listdir(output_dir):
        msg="Output directory ({}) already exists and is not empty.".format(output_dir)
        if input(msg+"=> overwrite?(Y/N)") not in ('Y','y'):
            raise ValueError(msg)

    os.makedirs(output_dir, exist_ok=True)

    task_name = task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](dataSource)
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    train_examples = None
    num_train_steps = None

    if do_train:
        train_examples = processor.get_train_examples(data_dir)
        num_train_steps = int(
            len(train_examples) / train_batch_size / gradient_accumulation_steps * num_train_epochs)

    # Prepare model
    model = BertForSemanticPrediction.from_pretrained(bert_model,
              num_labels = num_labels)

    if fp16:
        model.half()
    model.to(device)

    model = torch.nn.DataParallel(model,device_ids=[0,1])

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps

    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=t_total)
    tr_loss=None
    global_step = 0

    if do_train:
        train_features = SemanticRanker.convert_examples_to_features(
            train_examples, label_list, max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_sim_values=torch.tensor([f.simValue for f in train_features],dtype=torch.float)
        #print("train",all_sim_values.size(),all_label_ids.size(),all_segment_ids.size(),all_input_mask.size(),all_input_ids.size())

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_sim_values)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        model.train()
        for _ in trange(int(num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids,sim_values = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids,sim_values)
                loss=loss_partial[0]*loss[0]+loss_partial[1]*loss[1]
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = learning_rate * warmup_linear(global_step/t_total, warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)



    # Load a trained model that you have fine-tuned
    #model_state_dict = torch.load(output_model_file)
    #model = BertForSemanticPrediction.from_pretrained(bert_model, state_dict=model_state_dict,num_labels=num_labels)
    #model.to(device)

    sranker=SemanticRanker(output_model_file)

    if do_eval:
        eval_examples = processor.get_dev_examples(data_dir)
        eval_features = SemanticRanker.convert_examples_to_features(
            eval_examples, label_list, max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_sim_values=torch.tensor([f.simValue for f in eval_features],dtype=torch.float)

        #print("eval",all_sim_values.size(),all_label_ids.size(),all_segment_ids.size(),all_input_mask.size(),all_input_ids.size())
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_sim_values)
        test_data=TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        test_sampler=SequentialSampler(test_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
        test_dataloader=DataLoader(test_data,sampler=test_sampler,batch_size=eval_batch_size)
        out_logits,out_simValues =sranker.computeSimvalue(test_dataloader)
        print(out_simValues.shape,out_logits.shape)
        print(out_simValues[:3])
        print(out_logits[:3])
        print(all_sim_values[:3])

        model.eval()
        eval_loss, eval_accuracy,eval_error = 0, 0,0
        nb_eval_steps, nb_eval_examples = 0, 0
        i=0
        for input_ids, input_mask, segment_ids, label_ids,sim_values in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            sim_values=sim_values.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids,sim_values)
                #logits,simValues = model(input_ids, segment_ids, input_mask)
                tmp_eval_loss=loss_partial[0]*tmp_eval_loss[0]+loss_partial[1]*tmp_eval_loss[1]

            logits=out_logits[i:i+eval_batch_size]
            simValues=out_simValues[i:i+eval_batch_size]
            i+=eval_batch_size

            #logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            #simValues=simValues.detach().cpu().numpy()
            sim_values=sim_values.to('cpu').numpy()
            tmp_eval_error=mseError(simValues,sim_values)


            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            eval_error += tmp_eval_error

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        eval_error=eval_error/nb_eval_examples


        result = {'eval_loss': eval_loss,
                  'eval_error':eval_error,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': tr_loss/nb_tr_steps if tr_loss else "not availabel"
                  }

        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":

    dataSource=""
    main()
