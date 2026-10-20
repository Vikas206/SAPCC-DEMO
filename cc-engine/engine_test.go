package ccengine

import "testing"

func hotTxns() ([]string, []Transaction) {
	keys := AllKeyNames(200)
	hot := NewHotSetPredictor(20).Hot
	hotList := make([]string, 0, len(hot))
	for h := range hot {
		hotList = append(hotList, h)
	}
	txns := make([]Transaction, 0, 400)
	for i := 0; i < 400; i++ {
		txnKeys := append([]string(nil), hotList[:3]...)
		writes := []bool{true, true, false}
		txns = append(txns, Transaction{ID: itoa(i), Keys: txnKeys, Writes: writes,
			WorkNanos: 300_000, LockWaitNanos: 50_000})
	}
	return keys, txns
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	s := ""
	for i > 0 {
		s = string(rune('0'+i%10)) + s
		i /= 10
	}
	return s
}

func TestNoContentionAllStrategiesCommit(t *testing.T) {
	keys := AllKeyNames(100000)
	txns := make([]Transaction, 50)
	for i := range txns {
		txns[i] = Transaction{ID: itoa(i), Keys: []string{KeyName(i), KeyName(i + 50000)}, Writes: []bool{true, true}}
	}
	for _, s := range []CCStrategy{StaticOCC{}, StaticPessimistic{}, ReactiveSwitch{},
		&SAPCC{Pred: NewHotSetPredictor(5), Threshold: 0.5},
		&MLOnly{Pred: NewFrequencyPredictor(), Threshold: 0.5}} {
		sum := Run(s, keys, txns, 1) // single worker => zero contention
		if sum.Committed != sum.Total {
			t.Errorf("%s: %d/%d committed with no contention", s.Name(), sum.Committed, sum.Total)
		}
	}
}

func TestContentionBehaviour(t *testing.T) {
	keys, txns := hotTxns()
	occ := Run(StaticOCC{}, keys, txns, 16)
	pess := Run(StaticPessimistic{}, keys, txns, 16)
	rea := Run(ReactiveSwitch{}, keys, txns, 16)
	if occ.AbortRate == 0 {
		t.Error("static OCC should abort under hot-key contention")
	}
	if pess.Committed != pess.Total || rea.Committed != rea.Total {
		t.Error("pessimistic and reactive must always commit")
	}
	if pess.AbortRate != 0 {
		t.Error("pessimistic must never waste work")
	}
}

func TestSAPCCAvoidsWastedWorkAndLogs(t *testing.T) {
	keys, txns := hotTxns()
	log := &OutcomeLogger{}
	sap := &SAPCC{Pred: NewHotSetPredictor(20), Threshold: 0.25, Log: log}
	rea := Run(ReactiveSwitch{}, keys, txns, 16)
	got := Run(sap, keys, txns, 16)
	if got.Committed != got.Total {
		t.Errorf("sapcc committed %d/%d", got.Committed, got.Total)
	}
	if got.AbortRate >= rea.AbortRate {
		t.Errorf("sapcc wasted-work %.3f should be below reactive %.3f", got.AbortRate, rea.AbortRate)
	}
	if log.Len() != len(txns) {
		t.Errorf("expected %d outcome records, got %d", len(txns), log.Len())
	}
}

type failingPredictor struct{}

func (failingPredictor) Predict(*Transaction) (float64, error) { return 0, errTest }

var errTest = &testErr{}

type testErr struct{}

func (*testErr) Error() string { return "predictor down" }

func TestPredictorFailureFallsBackToPessimistic(t *testing.T) {
	keys, txns := hotTxns()
	got := Run(&SAPCC{Pred: failingPredictor{}, Threshold: 0.5}, keys, txns, 16)
	if got.Committed != got.Total || got.AbortRate != 0 {
		t.Errorf("fallback should commit everything with zero wasted work: %+v", got)
	}
}

func TestMLOnlyLearnsOnlineAndAvoidsWastedWork(t *testing.T) {
	keys, txns := hotTxns()
	log := &OutcomeLogger{}
	ml := &MLOnly{Pred: NewFrequencyPredictor(), Threshold: 0.25, Log: log}
	rea := Run(ReactiveSwitch{}, keys, txns, 16)
	got := Run(ml, keys, txns, 16)
	if got.Committed != got.Total {
		t.Errorf("ml_only committed %d/%d", got.Committed, got.Total)
	}
	// With zero prior knowledge, the online predictor must earn its routing
	// decisions from observed conflicts as it runs -- confirming it is not
	// just a disguised oracle -- yet across the full run should still beat
	// a baseline that never predicts at all.
	if got.AbortRate >= rea.AbortRate {
		t.Errorf("ml_only wasted-work %.3f should be below reactive %.3f", got.AbortRate, rea.AbortRate)
	}

	fresh := NewFrequencyPredictor()
	first := &Transaction{ID: "probe", Keys: []string{KeyName(1), KeyName(2)}, Writes: []bool{true, true}}
	before, _ := fresh.Predict(first)
	fresh.Observe(first, true) // simulate an observed conflict on these keys
	after, _ := fresh.Predict(first)
	if after <= before {
		t.Errorf("predicted conflict probability should rise after an observed conflict: before=%.3f after=%.3f", before, after)
	}
}

func TestPessimisticPaysLockWaitLatencyUnderNoContention(t *testing.T) {
	keys := AllKeyNames(100000)
	txns := make([]Transaction, 30)
	for i := range txns {
		txns[i] = Transaction{ID: itoa(i), Keys: []string{KeyName(i)}, Writes: []bool{true}, LockWaitNanos: 2_000_000}
	}
	occ := Run(StaticOCC{}, keys, txns, 1)
	pess := Run(StaticPessimistic{}, keys, txns, 1)
	if pess.P50ms <= occ.P50ms {
		t.Errorf("pessimistic p50 (%.2fms) should exceed OCC p50 (%.2fms) once lock-wait cost is modeled", pess.P50ms, occ.P50ms)
	}
}
