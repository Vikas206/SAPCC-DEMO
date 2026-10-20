package main

import (
	"flag"
	"fmt"
	"os"

	cc "sapcc/ccengine"
)

func main() {
	warehouses := flag.Int("warehouses", 20, "TPC-C warehouse count (also scales key space)")
	tpccTxns := flag.Int("tpcc-txns", 6000, "TPC-C transactions per strategy")
	ycsbTxns := flag.Int("ycsb-ops", 6000, "YCSB operation count per workload letter")
	ycsbRecords := flag.Int("ycsb-records", 10000, "YCSB recordcount")
	zipfS := flag.Float64("zipfian-s", 1.3, "Zipfian skew for YCSB workloads A/B/C/F")
	workers := flag.Int("workers", 16, "concurrent workers")
	workUs := flag.Int64("work-us", 150, "simulated txn processing time (microseconds)")
	lockUs := flag.Int64("lock-wait-us", 120, "simulated per-lock acquisition cost (microseconds) -- real cross-shard RTT stand-in")
	tau := flag.Float64("tau", 0.3, "SAPCC confidence threshold")
	predURL := flag.String("predictor-url", "", "semantic-service URL (empty = built-in oracle stand-in)")
	logPrefix := flag.String("outcome-log", "", "write SAPCC outcome records (JSONL) per workload, for Phase-2 training")
	flag.Parse()

	if *predURL == "" {
		fmt.Fprintln(os.Stderr, "note: -predictor-url not set; sapcc uses the Phase-1/3 oracle stand-in, not the real semantic-service")
	}

	applyCosts := func(txns []cc.Transaction) {
		for i := range txns {
			txns[i].WorkNanos = *workUs * 1000
			txns[i].LockWaitNanos = *lockUs * 1000
		}
	}

	// runSet gives every strategy its OWN OutcomeLogger: sharing one across
	// SAPCC and MLOnly would interleave SAPCC's semantic-labeled records with
	// MLOnly's non-semantic ones (empty SemanticText, conflicts observed
	// under a different routing policy) into a single Phase-2 training file.
	// Only SAPCC's own log is written -- that is the only one Phase 2 trains on.
	runSet := func(label string, keys []string, txns []cc.Transaction, pred cc.Predictor) {
		applyCosts(txns)
		if *predURL != "" {
			pred = cc.NewHTTPPredictor(*predURL)
		}
		sapccLog := &cc.OutcomeLogger{}
		sts := []cc.CCStrategy{cc.StaticOCC{}, cc.StaticPessimistic{}, cc.ReactiveSwitch{},
			&cc.SAPCC{Pred: pred, Threshold: *tau, Log: sapccLog},
			// Fresh, cold-start FrequencyPredictor per workload: it learns
			// conflict rates online from scratch (no zero-shot semantic
			// reasoning), the honest baseline SAPCC is measured against.
			&cc.MLOnly{Pred: cc.NewFrequencyPredictor(), Threshold: *tau, Log: &cc.OutcomeLogger{}},
		}
		for _, st := range sts {
			r := cc.Run(st, keys, txns, *workers)
			fmt.Printf("%-16s %-20s %8.1f%% %8.1f%% %9.0f %9.2f %9.2f\n", label, r.Strategy,
				100*r.AbortRate, 100*float64(r.Locked)/float64(r.Total), r.ThroughputTPS, r.P50ms, r.P99ms)
		}
		if *logPrefix != "" {
			if err := sapccLog.WriteJSONL(fmt.Sprintf("%s.%s.jsonl", *logPrefix, label)); err != nil {
				fmt.Fprintln(os.Stderr, "write log:", err)
			}
		}
	}

	fmt.Printf("%-16s %-20s %9s %9s %9s %9s %9s\n", "workload", "strategy", "abort%", "locked%", "tps", "p50ms", "p99ms")

	hotN := *ycsbRecords / 200
	ycsbKeys := cc.AllKeyNames(*ycsbRecords)
	for _, w := range []cc.YCSBWorkload{cc.YCSB_A, cc.YCSB_B, cc.YCSB_C, cc.YCSB_D, cc.YCSB_F} {
		txns := cc.GenerateYCSB(cc.YCSBConfig{Workload: w, RecordCount: *ycsbRecords, OperationCount: *ycsbTxns,
			OpsPerTxn: 4, ZipfS: *zipfS, Seed: 42})
		var pred cc.Predictor = cc.NewHotSetPredictor(hotN) // correct for A/B/C/F's low-index Zipfian skew
		if w == cc.YCSB_D {
			pred = &cc.LatestSetPredictor{RecordCount: *ycsbRecords, HotN: hotN} // D's hot keys are high-index, not low
		}
		runSet("YCSB-"+string(w), ycsbKeys, txns, pred)
	}

	tpccCfg := cc.TPCCConfig{Warehouses: *warehouses, NumTxns: *tpccTxns, Seed: 7}
	tpccKeys := cc.TPCCAllKeys(tpccCfg)
	txns := cc.GenerateTPCC(tpccCfg)
	runSet("TPCC", tpccKeys, txns, cc.TPCCSemanticPredictor{})
}
