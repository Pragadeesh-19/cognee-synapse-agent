"""
Phase B exit gate: verify state hash collision properties across all 60 questions.

Within-family (share-with-any): fraction of questions whose hash is shared by
at least one sibling of the same authored intent. Must exceed 60%.

Cross-family: fraction of cross-intent question pairs sharing a hash.
Must be below 10%.
"""
import json
from collections import defaultdict

from agent.state_hash import classify_intent, extract_tables, hash_state

QF = "benchmarks/questions.json"


def main():
    with open(QF, encoding="utf-8") as f:
        questions = json.load(f)

    # Run the live pipeline on every question text
    results = []
    for q in questions:
        detected_intent = classify_intent(q["question"])
        detected_tables = extract_tables(q["question"])
        h = hash_state({"intent": detected_intent, "tables": detected_tables, "clauses_so_far": []})
        results.append({
            "id": q["id"],
            "question": q["question"],
            "authored_intent": q["intent"],
            "authored_tables": q["tables"],
            "detected_intent": detected_intent,
            "detected_tables": detected_tables,
            "hash": h,
        })

    _print_classifier_accuracy(results)
    _print_collision_matrix(results)
    _print_cluster_breakdown(results)
    _print_gate(results)


def _print_classifier_accuracy(results):
    print("\n--- CLASSIFIER ACCURACY -------------------------------------------")
    intent_correct = sum(1 for r in results if r["detected_intent"] == r["authored_intent"])
    print(f"  Intent classification: {intent_correct}/{len(results)} = {intent_correct/len(results)*100:.1f}%")

    mismatches = [r for r in results if r["detected_intent"] != r["authored_intent"]]
    if mismatches:
        print("  Mismatches:")
        for r in mismatches:
            print(f"    Q{r['id']:02d}: authored={r['authored_intent']} detected={r['detected_intent']}")
            print(f"         {r['question']}")

    table_correct = 0
    table_mismatches = []
    for r in results:
        if sorted(r["detected_tables"]) == sorted(r["authored_tables"]):
            table_correct += 1
        else:
            table_mismatches.append(r)
    print(f"  Table extraction:      {table_correct}/{len(results)} = {table_correct/len(results)*100:.1f}%")
    if table_mismatches:
        print("  Mismatches:")
        for r in table_mismatches:
            print(f"    Q{r['id']:02d}: authored={r['authored_tables']} detected={r['detected_tables']}")
            print(f"         {r['question']}")


def _print_collision_matrix(results):
    print("\n--- WITHIN-FAMILY COLLISION (share-with-any) ----------------------")

    by_intent = defaultdict(list)
    for r in results:
        by_intent[r["authored_intent"]].append(r)

    family_rates = {}
    for intent, members in sorted(by_intent.items()):
        hashes = [r["hash"] for r in members]
        colliding = sum(1 for i, h in enumerate(hashes) if hashes.count(h) > 1)
        rate = colliding / len(members) if members else 0.0
        family_rates[intent] = rate
        print(f"  {intent:9s}: {colliding:2d}/{len(members):2d} collide = {rate*100:.1f}%")

    overall_colliding = sum(
        1 for r in results
        if sum(1 for r2 in results
               if r2["authored_intent"] == r["authored_intent"] and r2["hash"] == r["hash"]) > 1
    )
    overall_rate = overall_colliding / len(results)
    print(f"  {'OVERALL':9s}: {overall_colliding:2d}/{len(results):2d} collide = {overall_rate*100:.1f}%")

    print("\n--- CROSS-FAMILY COLLISION -----------------------------------------")
    cross_pairs = total_cross = 0
    intents = sorted(by_intent.keys())
    for i in range(len(intents)):
        for j in range(i + 1, len(intents)):
            a_hashes = {r["hash"] for r in by_intent[intents[i]]}
            b_hashes = {r["hash"] for r in by_intent[intents[j]]}
            shared = a_hashes & b_hashes
            n_pairs = len(by_intent[intents[i]]) * len(by_intent[intents[j]])
            colliding = sum(
                1 for ra in by_intent[intents[i]]
                for rb in by_intent[intents[j]]
                if ra["hash"] == rb["hash"]
            )
            cross_pairs += colliding
            total_cross += n_pairs
            if shared:
                print(f"  {intents[i]} x {intents[j]}: {colliding} colliding pairs (shared hashes: {shared})")
    cross_rate = cross_pairs / total_cross if total_cross else 0.0
    print(f"  Cross-family pair collision rate: {cross_pairs}/{total_cross} = {cross_rate*100:.2f}%")
    return overall_rate, cross_rate


def _print_cluster_breakdown(results):
    print("\n--- CLUSTER BREAKDOWN (intent x tables -> hash) -------------------")
    by_intent = defaultdict(list)
    for r in results:
        by_intent[r["authored_intent"]].append(r)

    for intent, members in sorted(by_intent.items()):
        clusters = defaultdict(list)
        for r in members:
            key = (r["detected_intent"], tuple(sorted(r["detected_tables"])))
            clusters[key].append(r["id"])
        print(f"  {intent}:")
        for (det_intent, det_tables), ids in sorted(clusters.items(), key=lambda x: -len(x[1])):
            marker = "" if det_intent == intent else f"  <-- MISCLASSIFIED as {det_intent}"
            print(f"    ({det_intent}, {list(det_tables)}): {len(ids)} questions -> Q{ids}{marker}")


def _print_gate(results):
    print("\n--- EXIT GATE ------------------------------------------------------")

    by_intent = defaultdict(list)
    for r in results:
        by_intent[r["authored_intent"]].append(r)

    within_colliding = sum(
        1 for r in results
        if sum(1 for r2 in results
               if r2["authored_intent"] == r["authored_intent"] and r2["hash"] == r["hash"]) > 1
    )
    within_rate = within_colliding / len(results)

    intents = sorted(by_intent.keys())
    cross_pairs = total_cross = 0
    for i in range(len(intents)):
        for j in range(i + 1, len(intents)):
            for ra in by_intent[intents[i]]:
                for rb in by_intent[intents[j]]:
                    total_cross += 1
                    if ra["hash"] == rb["hash"]:
                        cross_pairs += 1
    cross_rate = cross_pairs / total_cross if total_cross else 0.0

    within_pass = within_rate > 0.60
    cross_pass = cross_rate < 0.10
    print(f"  Within-family collision > 60%: {within_rate*100:.1f}%  {'PASS' if within_pass else 'FAIL'}")
    print(f"  Cross-family collision  < 10%: {cross_rate*100:.2f}%  {'PASS' if cross_pass else 'FAIL'}")
    print()
    if within_pass and cross_pass:
        print("  PASS -- Phase B exit gate satisfied")
    else:
        print("  FAIL -- Phase B exit gate not satisfied")
        if not within_pass:
            print("    Fix: improve classify_intent or re-cluster questions so same-intent questions share table-sets")
        if not cross_pass:
            print("    Fix: ensure different intents with same tables do not share hashes (check hash_state)")


if __name__ == "__main__":
    main()
