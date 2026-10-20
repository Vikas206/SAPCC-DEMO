package ccengine

import "time"

// SAPCC selects the concurrency-control strategy BEFORE execution:
//
//	P >= Threshold  -> pessimistic locking
//	P <  Threshold  -> optimistic, with a pessimistic safety-net retry on conflict
//	predictor error -> safe fallback to pessimistic (never trust a missing prediction)
type SAPCC struct {
	Pred      Predictor
	Threshold float64
	Log       *OutcomeLogger
}

func (s *SAPCC) Name() string { return "sapcc" }

func (s *SAPCC) Execute(store *Store, txn *Transaction) Result {
	start := time.Now() // includes prediction latency: the real cost of the approach
	p, err := s.Pred.Predict(txn)

	rec := OutcomeRecord{TxnID: txn.ID, SemanticType: txn.SemanticType, SemanticText: txn.SemanticText,
		NumKeys: len(txn.Keys), NumWrites: countTrue(txn.Writes), Predicted: p}

	if err != nil || p >= s.Threshold {
		rec.Route = "pessimistic"
		if err != nil {
			rec.Route = "fallback"
		}
		r := StaticPessimistic{}.Execute(store, txn)
		r.Strategy = "sapcc"
		r.LatencyNs = time.Since(start).Nanoseconds()
		s.Log.Add(rec)
		return r
	}

	rec.Route, rec.Labeled = "occ", true
	first := StaticOCC{}.Execute(store, txn)
	if first.Committed {
		s.Log.Add(rec)
		first.Strategy = "sapcc"
		first.LatencyNs = time.Since(start).Nanoseconds()
		return first
	}
	rec.Conflict = true
	s.Log.Add(rec)
	second := StaticPessimistic{}.Execute(store, txn)
	return Result{TxnID: txn.ID, Strategy: "sapcc", Committed: second.Committed, Retried: true,
		UsedLocking: true, LatencyNs: time.Since(start).Nanoseconds()}
}

func countTrue(b []bool) int {
	n := 0
	for _, v := range b {
		if v {
			n++
		}
	}
	return n
}
