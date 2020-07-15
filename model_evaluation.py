import random
import json
import MsmarcoQuestionAnswering.Evaluation.ms_marco_eval as eval_manager
import MsmarcoQuestionAnswering.Baseline.scripts.dataset as dataset_manager
import MsmarcoQuestionAnswering.Baseline.scripts.train as train_manager
import MsmarcoQuestionAnswering.Baseline.scripts.predict as predict_manager

class Evaluator:
    def __init__(self, config, data_path = "Dataset/dev_v2.1.json"):
        print("Load data for reference file")
        with open(data_path) as f_o:
                data_obj = json.loads(f_o.read())

        print("Create reference file")
        with open("correct_prediction.json", 'w+') as write_f:
            for qid in data_obj["answers"]:
                try:
                    correct = {"query_id": str(qid)}
                    correct["answers"] = data_obj["answers"][str(qid)]
                    write_f.write(json.dumps(correct))
                    write_f.write("\n")
                except KeyError:
                    print("Key Error: "+str(obj["query_id"]))
                
        print("Tokenize data")
        token_to_id = {'': 0}
        char_to_id = {'': 0}
        with open(data_path) as f_o:
            data, _ = dataset_manager.load_data(json.load(f_o), span_only=True, answered_only=True, loading_limit=1000)
        data = train_manager.tokenize_data(data, token_to_id, char_to_id)
        self.id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
        self.dev_data = train_manager.get_loader(data, config)

        print("Done.")

    def prep(self, model, id_to_token, data):
        qid2candidate = {}
        print("\t predict")
        for qid, toks, start, end in predict_manager.predict(model, data):
            toks = predict_manager.regex_multi_space.sub(' ', predict_manager.regex_drop_char.sub(' ', ' '.join(id_to_token[int(tok)] for tok in toks).lower())).strip()
            #print(repr(qid), repr(toks), start, end, file=f_o)
            #output = '{\"query_id\": '+ qid + ',\"answers\":[ \"' + toks + '\"]}'
            if qid not in qid2candidate:
                qid2candidate[qid] = []
            qid2candidate[qid].append(str(toks))
        print("\t no answer set")
        no_ans_set = set()
        for qid in qid2candidate:
            if len(qid2candidate[qid]) < 1 or 'No Answer Present.' in qid2candidate[qid]:
                no_ans_set.add(qid)
        print("\t take random answer from possible ones")
        out_dict = {}
        for qid in qid2candidate:
            pick = random.randint(0,len(qid2candidate[qid])-1)
            out_dict[qid] = [qid2candidate[qid][pick]]
        return out_dict, no_ans_set

    def eval(self, model):
        print("prepare evaluation")
        qid2ans_dict, no_ans_set = self.prep(model, self.id_to_token, self.dev_data)
        print("evaluate ...")
        metrics = eval_manager.compute_metrics_from_model("correct_prediction.json", qid2ans_dict, no_ans_set)
        return metrics
    
    def __call__(self, model):
        return self.eval(model)
