package ccengine

// TPC-C generator reproducing TPC Benchmark C, Standard Specification
// Revision 5.11, as extracted from the uploaded spec PDF and cross-checked
// against CMU BenchBase's production TPCCConfig.java / sample_tpcc_config.xml:
//
//   Schema cardinality (Clause 1.2, ER diagram):  W warehouses, 10
//   districts/warehouse, 3,000 customers/district, 100,000 items (global),
//   100,000 stock rows/warehouse.
//
//   Transaction mix: Clause 5.2.3 sets only MINIMUMS (Payment >=43.0%,
//   Order-Status/Delivery/Stock-Level >=4.0% each; New-Order has no minimum
//   -- "its measured rate is the reported throughput"). 45/43/4/4/4 is the
//   specific weighting BenchBase's sample_tpcc_config.xml ships for every
//   DB target (<weights>45,43,4,4,4</weights>), confirmed against the
//   uploaded benchbase-main.zip source, not a spec-mandated figure itself.
//
//   New-Order: ol_cnt randomly selected in [5..15] (Clause 2.4.1.3); for
//   each order-line, 99% home-warehouse / 1% remote-warehouse supply
//   (Clause 2.4.1.5, terms defined in 2.4.1.7/2.4.1.8); C_ID selected via
//   NURand(1023,1,3000) (Clause 2.4.1.2).
//   Payment: customer resident warehouse is home 85% / remote 15% of the
//   time; C_ID via NURand(1023,1,3000) (Clause 2.5.1.2 -- this generator
//   always takes the "by number" branch, not the 60%-by-C_LAST branch,
//   since C_LAST lookups aren't modelled at this key granularity).
//   NURand(A,x,y) = (((random(0,A) | random(x,y)) + C) % (y-x+1)) + x, with
//   C held fixed for the run (Clause 2.1.6): A=8191 for OL_I_ID in
//   [1,100000], A=1023 for C_ID in [1,3000] -- both verified character-for-
//   character against the uploaded spec PDF.
//
// Keys are modelled at row granularity: "W{w}" (warehouse), "W{w}D{d}"
// (district), "W{w}D{d}C{c}" (customer), "W{w}S{item}" (per-warehouse
// stock). New rows a transaction only inserts (Order/New-Order/Order-Line)
// are not existing keys and are excluded, since they cannot conflict with
// any other in-flight transaction.

import (
	"fmt"
	"math/rand"
)

type TPCCConfig struct {
	Warehouses int
	NumTxns    int
	Seed       int64
}

const (
	tpccDistPerWhse = 10
	tpccCustPerDist = 3000
	tpccItemCount   = 100000
)

// nurand implements Clause 2.1.6's NURand(A,x,y) exactly, with a per-run C
// held fixed for the duration of GenerateTPCC (spec 2.1.6: "the same C
// value ... must be used by all emulated terminals").
func nurand(r *rand.Rand, c, a, x, y int) int {
	rand1 := r.Intn(a + 1)
	rand2 := x + r.Intn(y-x+1)
	return ((rand1|rand2)+c)%(y-x+1) + x
}

func whKey(w int) string          { return fmt.Sprintf("W%d", w) }
func distKey(w, d int) string     { return fmt.Sprintf("W%dD%d", w, d) }
func custKey(w, d, c int) string  { return fmt.Sprintf("W%dD%dC%d", w, d, c) }
func stockKey(w, item int) string { return fmt.Sprintf("W%dS%d", w, item) }

// TPCCAllKeys enumerates every row key GenerateTPCC can reference, so the
// benchmark store is pre-populated with the real TPC-C schema's row set
// (Clause 1.2 cardinalities: 10 districts/warehouse, 3,000 customers/
// district, 100,000 stock rows/warehouse, plus the warehouse row itself).
func TPCCAllKeys(cfg TPCCConfig) []string {
	keys := make([]string, 0, cfg.Warehouses*(1+tpccDistPerWhse+tpccDistPerWhse*tpccCustPerDist+tpccItemCount))
	for w := 1; w <= cfg.Warehouses; w++ {
		keys = append(keys, whKey(w))
		for d := 1; d <= tpccDistPerWhse; d++ {
			keys = append(keys, distKey(w, d))
			for c := 1; c <= tpccCustPerDist; c++ {
				keys = append(keys, custKey(w, d, c))
			}
		}
		for item := 1; item <= tpccItemCount; item++ {
			keys = append(keys, stockKey(w, item))
		}
	}
	return keys
}

func GenerateTPCC(cfg TPCCConfig) []Transaction {
	r := rand.New(rand.NewSource(cfg.Seed))
	cItem := r.Intn(8192) // run-time C for OL_I_ID, range [0,8191]
	cCust := r.Intn(1024) // run-time C for C_ID, range [0,1023]

	pick := func(n int) int { return r.Intn(n) + 1 }
	remoteWh := func(home int) int {
		if cfg.Warehouses < 2 {
			return home
		}
		for {
			w := pick(cfg.Warehouses)
			if w != home {
				return w
			}
		}
	}

	txns := make([]Transaction, 0, cfg.NumTxns)
	for i := 0; i < cfg.NumTxns; i++ {
		w, d := pick(cfg.Warehouses), pick(tpccDistPerWhse)
		var keys []string
		var writes []bool
		var typ, text string

		// 45/43/4/4/4 deck mix (spec 5.2.3; matches BenchBase's production
		// <weights>45,43,4,4,4</weights> for the NoisePage/Postgres configs).
		switch roll := r.Float64() * 100; {
		case roll < 45: // New-Order
			typ = "NewOrder"
			// Clause 2.4.1.2: "The non-uniform random customer number (C_ID)
			// is selected using the NURand(1023,1,3000) function" -- caught
			// cross-checking against the uploaded spec PDF; this previously
			// used uniform pick(), which Payment's customer selection (below)
			// already did correctly.
			cust := nurand(r, cCust, 1023, 1, tpccCustPerDist)
			keys = append(keys, distKey(w, d)) // d_next_o_id increment: the classic TPC-C hotspot
			writes = append(writes, true)
			keys = append(keys, custKey(w, d, cust)) // read-only (discount/credit)
			writes = append(writes, false)
			nLines := 5 + r.Intn(11) // 5..15 per the ER diagram's "5-15" cardinality
			seen := map[string]bool{distKey(w, d): true, custKey(w, d, cust): true}
			for l := 0; l < nLines; l++ {
				item := nurand(r, cItem, 8191, 1, tpccItemCount)
				supplyW := w
				if r.Intn(100) == 0 { // 1% remote order-line (2.4.1.4/2.4.1.7)
					supplyW = remoteWh(w)
				}
				sk := stockKey(supplyW, item)
				if !seen[sk] {
					seen[sk] = true
					keys = append(keys, sk)
					writes = append(writes, true) // s_quantity decrement
				}
			}
			text = fmt.Sprintf("New-Order: writes district counter W%dD%d and %d stock row(s)", w, d, nLines)

		case roll < 88: // Payment (45+43)
			typ = "Payment"
			keys = append(keys, whKey(w), distKey(w, d)) // w_ytd, d_ytd aggregate counters
			writes = append(writes, true, true)
			custW, custD := w, d
			if r.Intn(100) < 15 { // 15% remote-warehouse customer (2.5.1.2)
				custW = remoteWh(w)
				custD = pick(tpccDistPerWhse)
			}
			cust := nurand(r, cCust, 1023, 1, tpccCustPerDist)
			keys = append(keys, custKey(custW, custD, cust)) // c_balance
			writes = append(writes, true)
			text = fmt.Sprintf("Payment: updates warehouse/district YTD and customer balance (remote=%v)", custW != w)

		case roll < 92: // Order-Status
			typ = "OrderStatus"
			cust := pick(tpccCustPerDist)
			keys = append(keys, custKey(w, d, cust))
			writes = append(writes, false) // read-only transaction
			text = "Order-Status: read-only customer + most recent order lookup"

		case roll < 96: // Delivery
			typ = "Delivery"
			text = "Delivery: processes the oldest undelivered order in each of the warehouse's 10 districts"
			for dd := 1; dd <= tpccDistPerWhse; dd++ {
				cust := pick(tpccCustPerDist)
				keys = append(keys, distKey(w, dd), custKey(w, dd, cust))
				writes = append(writes, true, true)
			}

		default: // Stock-Level
			typ = "StockLevel"
			for k := 0; k < 10; k++ { // approximates the ~20 most recent order-lines' items
				item := pick(tpccItemCount)
				keys = append(keys, stockKey(w, item))
				writes = append(writes, false)
			}
			text = "Stock-Level: read-only low-stock scan over recently ordered items"
		}

		txns = append(txns, Transaction{
			ID: fmt.Sprintf("tpcc-%d", i), Keys: keys, Writes: writes,
			SemanticType: typ, SemanticText: text,
		})
	}
	return txns
}
