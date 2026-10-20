package ccengine

import "time"

// MLOnly is the ForeSight-style ablation baseline: same pre-execution
// threshold-gated OCC-vs-pessimistic selection as SAPCC, but driven by
// FrequencyPredictor's raw per-key statistics instead of semantic intent,
// and updated online after every transaction instead of via Phase 2's
// offline calibration pass. Comparing MLOnly against SAPCC under identical
// workloads and an identical gating mechanism is what isolates the value
// of the semantic layer specifically -- the one variable that differs.
type MLOnly struct {
	Pred      *FrequencyPredictor
	Threshold float64
	Log       *OutcomeLogger
}

func (m *MLOnly) Name() string { return "ml_only_foresight" }

func (m *MLOnly) Execute(store *Store, txn *Transaction) Result {
	start := time.Now()
	p, _ := m.Pred.Predict(txn) // FrequencyPredictor.Predict never errors

	rec := OutcomeRecord{TxnID: txn.ID, SemanticType: txn.SemanticType,
		NumKeys: len(txn.Keys), NumWrites: countTrue(txn.Writes), Predicted: p}

	if p >= m.Threshold {
		rec.Route = "pessimistic"
		r := StaticPessimistic{}.Execute(store, txn)
		r.Strategy = "ml_only_foresight"
		r.LatencyNs = time.Since(start).Nanoseconds()
		m.Log.Add(rec)
		return r
	}

	rec.Route, rec.Labeled = "occ", true
	first := StaticOCC{}.Execute(store, txn)
	m.Pred.Observe(txn, !first.Committed) // online update, every transaction
	if first.Committed {
		m.Log.Add(rec)
		first.Strategy = "ml_only_foresight"
		first.LatencyNs = time.Since(start).Nanoseconds()
		return first
	}
	rec.Conflict = true
	m.Log.Add(rec)
	second := StaticPessimistic{}.Execute(store, txn)
	return Result{TxnID: txn.ID, Strategy: "ml_only_foresight", Committed: second.Committed,
		Retried: true, UsedLocking: true, LatencyNs: time.Since(start).Nanoseconds()}
}
