def evaluate(preds: list, gold_preds: list, mn_labels: list, verbose=False):
    stats = {}

    mn_correct = nmn_correct = 0
    for pred, gold, mn in zip(preds, gold_preds, mn_labels):
        if pred == gold:
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

    acc, mn_acc, nmn_acc = get_accuracies(preds, gold_preds, mn_labels, 'no_correct_tokens')
    stats["no_correct_tokens"] = {
        "acc" : acc,
        "mn_acc" : mn_acc,
        "nmn_acc" : nmn_acc
    }
    if verbose:
        print(f"NO. CORRECT TOKENS")
        print(f"ACC: {acc:.2f}%\nMN: {mn_acc:.2f}%\nNMN: {nmn_acc:.2f}%\n")

    acc, mn_acc, nmn_acc = get_accuracies(preds, gold_preds, mn_labels, 'max_correct_span')
    stats["max_correct_span"] = {
        "acc" : acc,
        "mn_acc" : mn_acc,
        "nmn_acc" : nmn_acc
    }
    if verbose:
        print(f"MAX CORRECT SPAN")
        print(f"ACC: {acc:.2f}%\nMN: {mn_acc:.2f}%\nNMN: {nmn_acc:.2f}%")
    
    return stats

def get_accuracies(preds: list, gold_preds: list, mn_labels: list, accuracy_type: str):
    acc = mn_acc = nmn_acc = 0
    lens = mn_lens = nmn_lens = 0
    for pred, gold, mn in zip(preds, gold_preds, mn_labels):
        pred = pred.split()
        gold = gold.split()
        tokens = 0
        for i in range(min(len(pred), len(gold))):
            if accuracy_type == 'no_correct_tokens':
                if pred[i] == gold[i]: tokens += 1
            elif accuracy_type == '':
                if pred[i] != gold[i]: continue
                correct = 1
                rhs_len = min(len(pred)-i-1, len(gold)-i-1)
                for j in range(rhs_len):
                    if pred[j] == gold[j]:
                        correct += 1
                    else: break
                if correct > tokens: tokens = correct                
        acc += (tokens * 100 / len(gold)) * len(gold)
        lens += len(gold)
        if mn: 
            mn_acc += (tokens * 100 / len(gold)) * len(gold)
            mn_lens += len(gold)
        else: 
            nmn_acc += (tokens * 100 / len(gold)) * len(gold)
            nmn_lens += len(gold)

    acc /= lens
    mn_acc /= mn_lens
    nmn_acc /= nmn_lens

    return acc, mn_acc, nmn_acc

