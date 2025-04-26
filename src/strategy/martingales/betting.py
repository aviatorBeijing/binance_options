import random

def blackjack_martingale_simulation(initial_bankroll, initial_bet, rounds):
    bankroll = initial_bankroll
    bet = initial_bet
    history = []

    for round_num in range(rounds):
        if bankroll <= 0:
            print("Bankroll depleted. Stopping the simulation.")
            break

        # Simulate the outcome (50% win chance)
        if random.random() < 0.5:  # Player wins
            bankroll += bet
            history.append(bankroll)
            bet = initial_bet  # Reset bet after a win
        else:  # Player loses
            bankroll -= bet
            history.append(bankroll)
            bet *= 2  # Double the bet after a loss

        # Check for bankroll limits (table limits could be added here)
        if bet > bankroll:
            bet = bankroll  # Bet cannot exceed current bankroll

    return history

# Parameters
initial_bankroll = 1000  # Starting money
initial_bet = 10         # Initial bet amount
rounds = 100             # Number of rounds to simulate

# Run the simulation
result_history = blackjack_martingale_simulation(initial_bankroll, initial_bet, rounds)

# Display results
print( result_history[-10:] )  # Show the last 10 bankroll states to see the end result

