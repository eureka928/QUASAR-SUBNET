#!/usr/bin/env python3
"""
QUASAR Subnet - Leaderboard Viewer

Fetches and displays the current leaderboard from the QUASAR validator API.

Usage:
    python scripts/leaderboard.py
    python scripts/leaderboard.py --top 10
    python scripts/leaderboard.py --league 1M
    python scripts/leaderboard.py --json
"""

import os
import sys
import json
import argparse
import requests
from datetime import datetime
from typing import Dict, List, Optional

# Validator API URL
VALIDATOR_API_URL = os.getenv("VALIDATOR_API_URL", "https://quasar-subnet.onrender.com")


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# League configuration (matches validator_api/app.py)
LEAGUE_CONFIG = {
    "1M":   {"min_seq": 1_000_000, "multiplier": 3.0,  "color": "\033[92m"},  # Green
    "900k": {"min_seq": 900_000,   "multiplier": 2.5,  "color": "\033[92m"},  # Green
    "800k": {"min_seq": 800_000,   "multiplier": 2.25, "color": "\033[93m"},  # Yellow
    "700k": {"min_seq": 700_000,   "multiplier": 2.0,  "color": "\033[93m"},  # Yellow
    "600k": {"min_seq": 600_000,   "multiplier": 1.75, "color": "\033[96m"},  # Cyan
    "500k": {"min_seq": 500_000,   "multiplier": 1.5,  "color": "\033[96m"},  # Cyan
    "400k": {"min_seq": 400_000,   "multiplier": 1.25, "color": "\033[0m"},   # Default
    "300k": {"min_seq": 300_000,   "multiplier": 1.0,  "color": "\033[0m"},   # Default
    "200k": {"min_seq": 200_000,   "multiplier": 0.75, "color": "\033[90m"},  # Gray
    "100k": {"min_seq": 100_000,   "multiplier": 0.5,  "color": "\033[90m"},  # Gray
}
RESET_COLOR = "\033[0m"


def get_league(seq_len: int) -> tuple:
    """Get league name, multiplier, and color for a sequence length."""
    for league_name, config in LEAGUE_CONFIG.items():
        if seq_len >= config["min_seq"]:
            return league_name, config["multiplier"], config["color"]
    return "Mini", 0.5, "\033[0m"


def fetch_submission_stats() -> Optional[Dict]:
    """Fetch submission statistics from validator API."""
    try:
        response = requests.get(
            f"{VALIDATOR_API_URL}/get_submission_stats",
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats: {e}")
        return None


def fetch_leaderboard() -> Optional[List[Dict]]:
    """Fetch leaderboard from validator API."""
    try:
        # Try dedicated leaderboard endpoint first
        response = requests.get(
            f"{VALIDATOR_API_URL}/leaderboard",
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass

    # Fall back to submission stats
    stats = fetch_submission_stats()
    if stats and "recent_submissions" in stats:
        return stats["recent_submissions"]
    return None


def calculate_weighted_score(submission: Dict) -> float:
    """Calculate weighted score for a submission."""
    tokens_per_sec = submission.get("tokens_per_sec", 0)
    seq_len = submission.get("target_sequence_length", 0)
    _, multiplier, _ = get_league(seq_len)
    return tokens_per_sec * multiplier


def display_leaderboard(submissions: List[Dict], top_n: int = 20, league_filter: str = None):
    """Display formatted leaderboard."""

    if not submissions:
        print("No submissions found.")
        return

    # Filter by league if specified
    if league_filter:
        league_filter = league_filter.upper()
        filtered = []
        for s in submissions:
            seq_len = s.get("target_sequence_length", 0)
            league, _, _ = get_league(seq_len)
            if league.upper() == league_filter:
                filtered.append(s)
        submissions = filtered

        if not submissions:
            print(f"No submissions found in {league_filter} league.")
            return

    # Calculate weighted scores and sort
    for s in submissions:
        s["weighted_score"] = calculate_weighted_score(s)
        seq_len = s.get("target_sequence_length", 0)
        s["league"], s["multiplier"], s["color"] = get_league(seq_len)

    # Sort by weighted score (descending)
    submissions.sort(key=lambda x: x["weighted_score"], reverse=True)

    # Limit to top N
    submissions = submissions[:top_n]

    # Display header
    print(f"{'Rank':<6} {'Miner':<20} {'League':<8} {'Tokens/s':<12} {'Mult':<6} {'Weighted Score':<15} {'Status'}")
    print("-" * 90)

    # Display entries
    for i, s in enumerate(submissions, 1):
        miner = s.get("miner_hotkey", "Unknown")[:18]
        league = s.get("league", "?")
        tokens = s.get("tokens_per_sec", 0)
        mult = s.get("multiplier", 1.0)
        weighted = s.get("weighted_score", 0)
        status = s.get("status", "pending")
        color = s.get("color", "")

        # Rank medal
        if i == 1:
            rank = "🥇 1"
        elif i == 2:
            rank = "🥈 2"
        elif i == 3:
            rank = "🥉 3"
        else:
            rank = f"   {i}"

        # Status indicator
        if status == "validated":
            status_str = "✓ Validated"
        elif status == "pending":
            status_str = "⏳ Pending"
        elif status == "failed":
            status_str = "✗ Failed"
        else:
            status_str = status

        print(f"{rank:<6} {miner:<20} {color}{league:<8}{RESET_COLOR} {tokens:>11,.0f} {mult:>5.1f}x {weighted:>14,.0f} {status_str}")

    print("-" * 90)
    print(f"Total: {len(submissions)} submissions shown")


def display_league_summary(submissions: List[Dict]):
    """Display summary by league."""

    if not submissions:
        return

    league_stats = {}

    for s in submissions:
        seq_len = s.get("target_sequence_length", 0)
        league, mult, color = get_league(seq_len)
        tokens = s.get("tokens_per_sec", 0)

        if league not in league_stats:
            league_stats[league] = {
                "count": 0,
                "best_tokens": 0,
                "best_weighted": 0,
                "multiplier": mult,
                "color": color,
            }

        league_stats[league]["count"] += 1
        if tokens > league_stats[league]["best_tokens"]:
            league_stats[league]["best_tokens"] = tokens
            league_stats[league]["best_weighted"] = tokens * mult

    print("\n" + "=" * 70)
    print("  League Summary")
    print("=" * 70 + "\n")

    print(f"{'League':<10} {'Mult':<8} {'Submissions':<12} {'Best Tok/s':<15} {'Best Weighted'}")
    print("-" * 65)

    for league in ["1M", "900k", "800k", "700k", "600k", "500k", "400k", "300k", "200k", "100k"]:
        if league in league_stats:
            stats = league_stats[league]
            color = stats["color"]
            print(f"{color}{league:<10}{RESET_COLOR} {stats['multiplier']:>6.1f}x {stats['count']:>10} {stats['best_tokens']:>14,.0f} {stats['best_weighted']:>14,.0f}")
        else:
            config = LEAGUE_CONFIG.get(league, {})
            mult = config.get("multiplier", 1.0)
            print(f"{league:<10} {mult:>6.1f}x {0:>10} {'-':>14} {'-':>14}")

    print("-" * 65)


def display_your_position(submissions: List[Dict], your_score: float):
    """Show where your score would rank."""

    # Calculate weighted scores
    for s in submissions:
        s["weighted_score"] = calculate_weighted_score(s)

    # Sort by weighted score
    submissions.sort(key=lambda x: x["weighted_score"], reverse=True)

    # Find position
    position = 1
    for s in submissions:
        if s["weighted_score"] > your_score:
            position += 1
        else:
            break

    total = len(submissions)

    print("\n" + "=" * 70)
    print("  Your Position")
    print("=" * 70 + "\n")

    if position == 1:
        print(f"  🏆 Your score ({your_score:,.0f}) would be #1!")
    elif position <= 4:
        print(f"  🎯 Your score ({your_score:,.0f}) would be #{position} (Top 4 = rewards!)")
    else:
        print(f"  📊 Your score ({your_score:,.0f}) would be #{position} out of {total}")

        # Show how much more needed for top 4
        if total >= 4:
            fourth_place = submissions[3]["weighted_score"]
            gap = fourth_place - your_score
            if gap > 0:
                print(f"  📈 Need +{gap:,.0f} weighted score to reach top 4")


def main():
    parser = argparse.ArgumentParser(description="QUASAR Leaderboard Viewer")
    parser.add_argument("--top", type=int, default=20, help="Show top N entries")
    parser.add_argument("--league", help="Filter by league (2M, 1.5M, 1M, 512K, 124K, 32K)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--your-score", type=float, help="Show where your score would rank")
    parser.add_argument("--api-url", help="Override validator API URL")
    args = parser.parse_args()

    global VALIDATOR_API_URL
    if args.api_url:
        VALIDATOR_API_URL = args.api_url

    print_header("QUASAR Subnet Leaderboard")
    print(f"API: {VALIDATOR_API_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Fetch data
    print("\nFetching leaderboard data...")
    submissions = fetch_leaderboard()

    if submissions is None:
        print("Failed to fetch leaderboard data.")

        # Try to get any stats
        stats = fetch_submission_stats()
        if stats:
            print("\nAPI Stats:")
            print(json.dumps(stats, indent=2))
        return

    if args.json:
        print(json.dumps(submissions, indent=2))
        return

    # Display leaderboard
    print_header("Current Standings")
    display_leaderboard(submissions, args.top, args.league)

    # Display league summary
    display_league_summary(submissions)

    # Show your position if specified
    if args.your_score:
        display_your_position(submissions, args.your_score)

    # Reward info
    print("\n" + "=" * 70)
    print("  Reward Distribution")
    print("=" * 70)
    print("""
  Top 4 miners share rewards:
    🥇 1st place: 60%
    🥈 2nd place: 25%
    🥉 3rd place: 10%
    4th place: 5%

  League Multipliers:
    1M:   3.0x  |  900k: 2.5x  |  800k: 2.25x |  700k: 2.0x
    600k: 1.75x |  500k: 1.5x  |  400k: 1.25x |  300k: 1.0x
    200k: 0.75x |  100k: 0.5x
""")


if __name__ == "__main__":
    main()
