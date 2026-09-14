import type { Dispatch, SetStateAction } from "react";
import { CANONICAL_BROKER_TIMEFRAMES } from "../timeframes";

/** Slowest-first display order, derived only from exact broker periods. */
export const TF_ORDER: string[] = [...CANONICAL_BROKER_TIMEFRAMES].reverse();

export const tfRank = (timeframe: string) => {
  const index = TF_ORDER.indexOf(timeframe);
  return index < 0 ? 999 : index;
};

/** Stable local timestamp that remains sortable at a glance. */
export const stamp = (ms: number | null | undefined) => {
  if (!ms) return "—";
  const date = new Date(ms);
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
};

export const ago = (ms: number | null | undefined) => {
  if (!ms) return "";
  const seconds = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
};

export const toggleIn =
  (set: Dispatch<SetStateAction<string[]>>) => (value: string) =>
    set((current) => current.includes(value)
      ? current.filter((candidate) => candidate !== value)
      : [...current, value]);
