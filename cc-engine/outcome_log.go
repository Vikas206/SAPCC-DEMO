package ccengine

import (
	"encoding/json"
	"os"
	"sync"
)

// OutcomeRecord is one closed-loop training example. Conflict is only
// observable (Labeled=true) when the optimistic path actually ran.
type OutcomeRecord struct {
	TxnID        string  `json:"txn_id"`
	SemanticType string  `json:"type"`
	SemanticText string  `json:"text"`
	NumKeys      int     `json:"num_keys"`
	NumWrites    int     `json:"num_writes"`
	Predicted    float64 `json:"predicted"`
	Route        string  `json:"route"` // "occ" | "pessimistic" | "fallback"
	Labeled      bool    `json:"labeled"`
	Conflict     bool    `json:"conflict"`
}

type OutcomeLogger struct {
	mu   sync.Mutex
	recs []OutcomeRecord
}

func (l *OutcomeLogger) Add(r OutcomeRecord) {
	if l == nil {
		return
	}
	l.mu.Lock()
	l.recs = append(l.recs, r)
	l.mu.Unlock()
}

func (l *OutcomeLogger) Len() int { l.mu.Lock(); defer l.mu.Unlock(); return len(l.recs) }

func (l *OutcomeLogger) WriteJSONL(path string) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, r := range l.recs {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return nil
}
