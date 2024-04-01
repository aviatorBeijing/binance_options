import random

def gen_random_sets(istart, iend, length, n):
    """
    @brief Generate <n> random subsets of <length> consequtive integers in range (istart, iend)
    ( Following algo is generated by ChatGPT 3.5 )
    """
    start_range = istart
    end_range = iend
    subset_length = length
    num_subsequences = n

    # Generate 5 random consecutive subsequences
    subsequences = []
    for _ in range(num_subsequences):
        # Choose a random starting point within a valid range
        start_point = random.randint(start_range, end_range - subset_length)
        
        # Generate the consecutive subsequence of length 30 starting from the chosen starting point
        subsequence = list(range(start_point, start_point + subset_length))
        
        # Append the subsequence to the list of subsequences
        subsequences.append(subsequence)

    ss = []
    for i, subsequence in enumerate(subsequences, 1):
       ss += [ subsequence ]
    return ss
