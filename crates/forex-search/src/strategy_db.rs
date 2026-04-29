use crate::genetic::strategy_gene::Gene;
use duckdb::{Connection, Result, params};
use tracing::info;

pub struct StrategyDb {
    conn: Connection,
}

impl StrategyDb {
    pub fn new(path: Option<&str>) -> Result<Self> {
        let conn = if let Some(p) = path {
            Connection::open(p)?
        } else {
            Connection::open_in_memory()?
        };

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS strategies (
                run_id TEXT,
                symbol TEXT,
                timeframe TEXT,
                strategy_id TEXT,
                indices TEXT,
                weights TEXT,
                long_threshold FLOAT,
                short_threshold FLOAT,
                fitness FLOAT,
                sharpe_ratio FLOAT,
                win_rate FLOAT,
                profit_factor FLOAT,
                max_drawdown FLOAT,
                trades_count INTEGER,
                use_ob BOOLEAN,
                use_fvg BOOLEAN,
                use_liq_sweep BOOLEAN,
                mtf_confirmation BOOLEAN,
                use_premium_discount BOOLEAN,
                use_inducement BOOLEAN,
                use_bos BOOLEAN,
                use_choch BOOLEAN,
                use_eqh BOOLEAN,
                use_eql BOOLEAN,
                use_displacement BOOLEAN,
                tp_pips FLOAT,
                sl_pips FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_strategies_symbol_tf ON strategies (symbol, timeframe);
            CREATE INDEX IF NOT EXISTS idx_strategies_id ON strategies (strategy_id);",
        )?;

        info!("StrategyDb initialized (path: {:?})", path);
        Ok(Self { conn })
    }

    pub fn insert_batch(&self, run_id: &str, symbol: &str, tf: &str, genes: &[Gene]) -> Result<()> {
        let mut stmt = self.conn.prepare(
            "INSERT INTO strategies (
                run_id, symbol, timeframe, strategy_id, indices, weights,
                long_threshold, short_threshold, fitness, sharpe_ratio,
                win_rate, profit_factor, max_drawdown, trades_count,
                use_ob, use_fvg, use_liq_sweep, mtf_confirmation,
                use_premium_discount, use_inducement, use_bos, use_choch,
                use_eqh, use_eql, use_displacement, tp_pips, sl_pips
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )?;

        for gene in genes {
            stmt.execute(params![
                run_id,
                symbol,
                tf,
                gene.strategy_id,
                serde_json::to_string(&gene.indices).unwrap_or_default(),
                serde_json::to_string(&gene.weights).unwrap_or_default(),
                gene.long_threshold,
                gene.short_threshold,
                gene.fitness,
                gene.sharpe_ratio,
                gene.win_rate,
                gene.profit_factor,
                gene.max_drawdown,
                gene.trades_count as i32,
                gene.use_ob,
                gene.use_fvg,
                gene.use_liq_sweep,
                gene.mtf_confirmation,
                gene.use_premium_discount,
                gene.use_inducement,
                gene.use_bos,
                gene.use_choch,
                gene.use_eqh,
                gene.use_eql,
                gene.use_displacement,
                gene.tp_pips,
                gene.sl_pips
            ])?;
        }

        info!(
            "Inserted {} strategies into DB (run: {}, symbol: {}, tf: {})",
            genes.len(),
            run_id,
            symbol,
            tf
        );
        Ok(())
    }

    pub fn cross_tf_winners(
        &self,
        min_tfs: usize,
        min_sharpe: f64,
    ) -> Result<Vec<(String, usize)>> {
        let mut stmt = self.conn.prepare(
            "SELECT strategy_id, COUNT(DISTINCT timeframe) as tf_count
             FROM strategies
             WHERE sharpe_ratio >= ?
             GROUP BY strategy_id
             HAVING tf_count >= ?
             ORDER BY tf_count DESC",
        )?;

        let rows = stmt.query_map(params![min_sharpe, min_tfs as i32], |row| {
            let id: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((id, count as usize))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn seed_population(&self, symbol: &str, limit: usize) -> Result<Vec<Gene>> {
        let mut stmt = self.conn.prepare(
            "SELECT 
                strategy_id, indices, weights, long_threshold, short_threshold,
                fitness, sharpe_ratio, win_rate, max_drawdown, profit_factor,
                trades_count, use_ob, use_fvg, use_liq_sweep, mtf_confirmation,
                use_premium_discount, use_inducement, use_bos, use_choch,
                use_eqh, use_eql, use_displacement, tp_pips, sl_pips
             FROM (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY strategy_id ORDER BY fitness DESC) as rn
                FROM strategies
                WHERE symbol = ?
             )
             WHERE rn = 1
             ORDER BY fitness DESC
             LIMIT ?",
        )?;

        let rows = stmt.query_map(params![symbol, limit as i32], |row| {
            let indices_s: String = row.get(1)?;
            let weights_s: String = row.get(2)?;
            Ok(Gene {
                strategy_id: row.get(0)?,
                indices: serde_json::from_str(&indices_s).unwrap_or_default(),
                weights: serde_json::from_str(&weights_s).unwrap_or_default(),
                long_threshold: row.get(3)?,
                short_threshold: row.get(4)?,
                fitness: row.get(5)?,
                sharpe_ratio: row.get(6)?,
                win_rate: row.get(7)?,
                max_drawdown: row.get(8)?,
                profit_factor: row.get(9)?,
                trades_count: row.get::<_, i32>(10)? as usize,
                use_ob: row.get(11)?,
                use_fvg: row.get(12)?,
                use_liq_sweep: row.get(13)?,
                mtf_confirmation: row.get(14)?,
                use_premium_discount: row.get(15)?,
                use_inducement: row.get(16)?,
                use_bos: row.get(17)?,
                use_choch: row.get(18)?,
                use_eqh: row.get(19)?,
                use_eql: row.get(20)?,
                use_displacement: row.get(21)?,
                tp_pips: row.get(22)?,
                sl_pips: row.get(23)?,
                ..Default::default()
            })
        })?;

        let mut genes = Vec::new();
        for row in rows {
            genes.push(row?);
        }

        if !genes.is_empty() {
            info!(
                "Seeding population with {} strategies from DB for symbol {}",
                genes.len(),
                symbol
            );
        }

        Ok(genes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::strategy_gene::Gene;

    #[test]
    fn test_strategy_db_lifecycle() -> Result<()> {
        let db = StrategyDb::new(None)?;
        let gene = Gene {
            strategy_id: "test_strat".to_string(),
            fitness: 100.0,
            sharpe_ratio: 1.5,
            indices: vec![1, 2, 3],
            weights: vec![0.5, 0.5, 0.5],
            ..Gene::default()
        };

        db.insert_batch("run1", "EURUSD", "M15", std::slice::from_ref(&gene))?;

        // Add another timeframe for same strategy
        db.insert_batch("run1", "EURUSD", "H1", std::slice::from_ref(&gene))?;

        let winners = db.cross_tf_winners(2, 1.0)?;
        assert_eq!(winners.len(), 1);
        assert_eq!(winners[0].0, "test_strat");
        assert_eq!(winners[0].1, 2);

        let seed = db.seed_population("EURUSD", 10)?;
        assert_eq!(seed.len(), 1);
        assert_eq!(seed[0].strategy_id, "test_strat");
        assert_eq!(seed[0].indices, vec![1, 2, 3]);

        Ok(())
    }
}
