def render_vs(env, v_star):
  policy_str = '-'*26 + '\n'
  for i in range(env.rows):
    for j in range(env.cols):
      action = round(v_star[(i, j)],3) if (i,j) in v_star and v_star[(i, j)] is not None else 'x'
      policy_str +=  str(action) + '\t'
    policy_str += '\n'
  policy_str += '-' * 26 + '\n'
  print(policy_str)

def render_policy(env, policy):
  policy_str = '-'*26 + '\n'
  for i in range(env.rows):
    for j in range(env.cols):
      action = policy[(i, j)][0][0][0] if (i,j) in policy and policy[(i, j)] is not None else 'x'
      policy_str +=  action + '\t'
    policy_str += '\n'
  policy_str += '-' * 26 + '\n'

  print(policy_str)
