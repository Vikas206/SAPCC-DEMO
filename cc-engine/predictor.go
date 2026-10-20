package ccengine

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Predictor returns the probability that txn will conflict.
type Predictor interface {
	Predict(txn *Transaction) (float64, error)
}

// HotSetPredictor is an ORACLE stand-in for engine tests / Phase-1 sanity runs:
// probability = share of the txn's keys that are in a known hot set.
// It is NOT the patent mechanism — see HTTPPredictor for the real LLM+calibration path.
type HotSetPredictor struct{ Hot map[string]bool }

func NewHotSetPredictor(hotN int) *HotSetPredictor {
	h := make(map[string]bool, hotN)
	for i := 0; i < hotN; i++ {
		h[KeyName(i)] = true
	}
	return &HotSetPredictor{Hot: h}
}

func (p *HotSetPredictor) Predict(txn *Transaction) (float64, error) {
	if len(txn.Keys) == 0 {
		return 0, nil
	}
	n := 0
	for _, k := range txn.Keys {
		if p.Hot[k] {
			n++
		}
	}
	return float64(n) / float64(len(txn.Keys)), nil
}

// HTTPPredictor calls the Python semantic-service (LLM intent -> calibrated probability).
type HTTPPredictor struct {
	URL    string
	Client *http.Client
}

func NewHTTPPredictor(url string) *HTTPPredictor {
	return &HTTPPredictor{URL: url, Client: &http.Client{Timeout: 2 * time.Second}}
}

type classifyReq struct {
	TxnID  string   `json:"txn_id"`
	Type   string   `json:"type"`
	Text   string   `json:"text"`
	Keys   []string `json:"keys"`
	Writes []bool   `json:"writes"`
}
type classifyResp struct {
	ConflictProbability float64 `json:"conflict_probability"`
	Confidence          float64 `json:"confidence"`
}

func (h *HTTPPredictor) Predict(txn *Transaction) (float64, error) {
	body, _ := json.Marshal(classifyReq{txn.ID, txn.SemanticType, txn.SemanticText, txn.Keys, txn.Writes})
	resp, err := h.Client.Post(h.URL, "application/json", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("classify: status %d", resp.StatusCode)
	}
	var out classifyResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, err
	}
	return out.ConflictProbability, nil
}
