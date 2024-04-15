def evaluate(preds: list, gold_preds: list, mn_labels: list, verbose=False):
    stats = {}

    mn_correct = nmn_correct = 0
    for pred, gold_pred, mn in zip(preds, gold_preds, mn_labels):
        if pred == gold_pred:
            if mn: mn_correct += 1
            else: nmn_correct += 1
        
    acc = mn_correct + nmn_correct
    acc *= 100
    acc /= len(preds)

    mn_acc = mn_correct * 100
    mn_acc /= sum(mn_labels)

    nmn_acc = nmn_correct * 100
    nmn_acc /= len(mn_labels) - sum(mn_labels)

    stats["exact_match"] = {
        "acc" : acc,
        "mn_acc" : mn_acc,
        "nmn_acc" : nmn_acc
    }
    if verbose:
        print(f"EXACT-MATCH ACCURACY")
        print(f"ACC: {acc:.2f}%\nMN: {mn_acc:.2f}%\nNMN: {nmn_acc:.2f}%\n")

    acc = mn_acc = nmn_acc = 0
    lens = mn_lens = nmn_lens = 0
    for pred, gold_pred, mn in zip(preds, gold_preds, mn_labels):
        pred = pred.split()
        gold_pred = gold_pred.split()

        correct_tokens = 0
        for i in range(min(len(pred), len(gold_pred))):
            if pred[i] == gold_pred[i]: correct_tokens += 1
        
        acc += (correct_tokens * 100 / len(gold_pred)) * len(gold_pred)
        lens += len(gold_pred)
        if mn: 
            mn_acc += (correct_tokens * 100 / len(gold_pred)) * len(gold_pred)
            mn_lens += len(gold_pred)
        else: 
            nmn_acc += (correct_tokens * 100 / len(gold_pred)) * len(gold_pred)
            nmn_lens += len(gold_pred)

    acc /= lens
    mn_acc /= mn_lens
    nmn_acc /= nmn_lens
    stats["no_correct_tokens"] = {
        "acc" : acc,
        "mn_acc" : mn_acc,
        "nmn_acc" : nmn_acc
    }
    if verbose:
        print(f"NO. CORRECT TOKENS")
        print(f"ACC: {acc:.2f}%\nMN: {mn_acc:.2f}%\nNMN: {nmn_acc:.2f}%\n")

    acc = mn_acc = nmn_acc = 0
    lens = mn_lens = nmn_lens = 0
    for pred, gold_pred, mn in zip(preds, gold_preds, mn_labels):
        pred = pred.split()
        gold_pred = gold_pred.split()

        max_correct = 0
        for i in range(min(len(pred), len(gold_pred))):
            if pred[i] != gold_pred[i]: continue
            correct = 1
            rhs_len = min(len(pred)-i-1, len(gold_pred)-i-1)
            for j in range(rhs_len):
                if pred[j] == gold_pred[j]:
                    correct += 1
                else: break
            if correct > max_correct: max_correct = correct
        
        acc += (max_correct * 100 / len(gold_pred)) * len(gold_pred)
        lens += len(gold_pred)
        if mn: 
            mn_acc += (max_correct * 100 / len(gold_pred)) * len(gold_pred)
            mn_lens += len(gold_pred)
        else: 
            nmn_acc += (max_correct * 100 / len(gold_pred)) * len(gold_pred)
            nmn_lens += len(gold_pred)
        
    acc /= lens
    mn_acc /= mn_lens
    nmn_acc /= nmn_lens

    stats["max_correct_span"] = {
        "acc" : acc,
        "mn_acc" : mn_acc,
        "nmn_acc" : nmn_acc
    }
    if verbose:
        print(f"MAX CORRECT SPAN")
        print(f"ACC: {acc:.2f}%\nMN: {mn_acc:.2f}%\nNMN: {nmn_acc:.2f}%")
    
    return stats