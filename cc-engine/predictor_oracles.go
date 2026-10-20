package ccengine

import "strconv"

// These two oracles are Phase-1/3 stand-ins for the real semantic-service
// (Phase 2). They let the engine and workload generators be validated end
// to end before the learned model exists. Neither is the patent mechanism.

// LatestSetPredictor is YCSB workload D's matching oracle: workload D's
// "latest" distribution biases toward recently inserted (high-index) keys,
// the opposite of A/B/C/F's low-index Zipfian skew, so it needs its own
// oracle rather than HotSetPredictor's low-index assumption.
type LatestSetPredictor struct {
	RecordCount, HotN int
}

func (p *LatestSetPredictor) Predict(txn *Transaction) (float64, error) {
	if len(txn.Keys) == 0 {
		return 0, nil
	}
	n := 0
	for _, k := range txn.Keys {
		// KeyName(i) == "key"+strconv.Itoa(i): recover i by parsing, not by
		// scanning all RecordCount keys -- this used to be O(RecordCount)
		// per key (a full linear rescan on every single prediction call),
		// which made YCSB-D benchmark runs needlessly slow at realistic
		// record counts. O(1) via strconv is both correct and the actual
		// cheap path the old comment claimed but didn't implement.
		idx, err := strconv.Atoi(k[len("key"):])
		if err != nil {
			continue // not a YCSB numeric key; can't be "recent" under this oracle
		}
		if idx >= p.RecordCount-p.HotN {
			n++
		}
	}
	return float64(n) / float64(len(txn.Keys)), nil
}

// TPCCSemanticPredictor stands in for "what the LLM should learn to
// recognize": conflict probability driven by the transaction's semantic
// TYPE, not by matching specific key strings. New-Order and Payment always
// write a shared per-district counter (the real TPC-C hotspot per Clause
// 2.4.2/2.5.1); Delivery writes all 10 district counters in one
// transaction; Order-Status and Stock-Level are read-only. A hot-key
// lookup like HotSetPredictor structurally cannot make this distinction
// for TPC-C's structured keys -- this is exactly the gap semantic
// classification is meant to fill, and Phase 2's logistic+Platt model
// should learn probabilities close to these from the outcome log rather
// than have them hardcoded.
type TPCCSemanticPredictor struct{}

func (TPCCSemanticPredictor) Predict(txn *Transaction) (float64, error) {
	switch txn.SemanticType {
	case "NewOrder":
		return 0.55, nil
	case "Payment":
		return 0.50, nil
	case "Delivery":
		return 0.70, nil
	default: // OrderStatus, StockLevel: read-only, cannot conflict
		return 0.0, nil
	}
}
