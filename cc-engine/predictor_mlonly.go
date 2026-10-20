package ccengine

import "sync"

// FrequencyPredictor is the ablation baseline standing in for ForeSight
// (Huang et al. 2025, arXiv:2508.17375): a lightweight, incrementally
// maintainable conflict predictor built ONLY from raw per-key access/
// conflict statistics -- no semantic text, no LLM, no offline calibration
// pass. Its purpose in this project is narrow: isolate whether the
// SEMANTIC layer (Phase 2's LLM intent classifier) adds predictive value
// beyond what raw statistics already capture, which is exactly the
// differentiation claim in the literature review ("ForeSight ... predicts
// from raw predicates/statistics, not from what the transaction is
// actually trying to do"). The real ASPN uses a sum-product network over
// access patterns; this uses a simpler per-key conflict-rate estimate with
// Bayesian (Beta-prior) smoothing, which is a faithful (if simplified)
// analogue of "predict from statistics alone" for a controlled ablation --
// it is not a reimplementation of ASPN and the project does not claim it is.
type FrequencyPredictor struct {
	mu       sync.Mutex
	access   map[string]int64
	conflict map[string]int64
}

func NewFrequencyPredictor() *FrequencyPredictor {
	return &FrequencyPredictor{access: map[string]int64{}, conflict: map[string]int64{}}
}

// Beta(priorA, priorB) prior per key: mean = priorA/(priorA+priorB) = 5%,
// effective sample size = priorA+priorB = 10 pseudo-observations. A
// symmetric Laplace/Jeffreys-style prior (a==b) always has mean 0.5
// regardless of its magnitude -- only an ASYMMETRIC prior can start an
// unseen key off "probably safe" rather than "coin flip," which matters
// here because transactions touch several keys at once.
const (
	priorA = 0.5
	priorB = 9.5
)

// Predict reports the touched key with the highest observed conflict
// rate, not an independent-union product over all keys. A per-key
// independent-union combination (P = 1 - prod_k(1-rate_k)) compounds even
// a small per-key rate into near-certainty once a transaction touches more
// than a handful of keys -- so a 17-key TPC-C Delivery transaction would
// always look dangerous purely from its key COUNT, never from which keys
// it actually touches. Taking the max keeps the score tied to the riskiest
// key regardless of transaction width, which is the property an
// incrementally-learned, per-key statistic should have.
func (p *FrequencyPredictor) Predict(txn *Transaction) (float64, error) {
	if len(txn.Keys) == 0 {
		return 0, nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	worst := 0.0
	for _, k := range txn.Keys {
		a, c := p.access[k], p.conflict[k]
		rate := (float64(c) + priorA) / (float64(a) + priorA + priorB)
		if rate > worst {
			worst = rate
		}
	}
	return worst, nil
}

// Observe updates the online statistics from one actually-executed
// optimistic attempt. This is the "incrementally maintainable" part of the
// ForeSight comparison: unlike SAPCC's calibration model, which retrains
// in a separate offline pass over the logged JSONL (Phase 2), this baseline
// updates immediately, in the hot path, on every transaction.
func (p *FrequencyPredictor) Observe(txn *Transaction, conflicted bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, k := range txn.Keys {
		p.access[k]++
		if conflicted {
			p.conflict[k]++
		}
	}
}
