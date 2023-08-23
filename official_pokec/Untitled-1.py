def cal_prob():
    dp=[0]*51
    dp[0]=1
    dp[1]=0.5
    #dp[2]=0.25
    for i in range(2,51):
        dp[i]=0.5*dp[i-2]+0.5*dp[i-1]
    return dp[50]

print(cal_prob())