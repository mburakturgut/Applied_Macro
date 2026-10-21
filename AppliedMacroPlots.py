"""Plotting functions for the two-period consumption-saving examples."""
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['PlotBudgetSet', 'PlotIndifferenceCurves', 'SolveUtilityMaximizationProblem', 'SolveUtilityMaximizationLagrange', 'InterpretLagrangeMultiplier']

def _budget(y_0, y_1, r):
    if r <= -1:
        raise ValueError('The gross interest rate must be positive.')
    wealth = y_0 + y_1/(1+r)
    if wealth <= 0:
        raise ValueError('Present-value wealth must be positive.')
    return wealth


def PlotBudgetSet(y_0, y_1, r, full_labels=True):
    wealth = _budget(y_0, y_1, r)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.linspace(0, wealth, 300)
    y = (1+r)*(wealth-x)
    ax.fill_between(x, 0, y, alpha=.15)
    ax.plot(x, y, label='Budget constraint')
    ax.scatter([y_0], [y_1], color='red', label='Income endowment')
    ax.set(xlim=(0, wealth*1.1), ylim=(0, (1+r)*wealth*1.1),
           xlabel='First-period consumption $c_0$', ylabel='Second-period consumption $c_1$',
           title='Two-period budget set')
    ax.legend()
    return ax


def PlotIndifferenceCurves(y_0, y_1, beta, r, levels=None, full_labels=True):
    if not 0 < beta < 1:
        raise ValueError('Use 0 < beta < 1.')
    wealth = _budget(y_0, y_1, r)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.linspace(wealth/1000, wealth*1.1, 600)
    for utility in ([0, .5, 1, 1.5] if levels is None else levels):
        y = np.exp((utility-np.log(x))/beta)
        ax.plot(x, y, label=f'Utility = {utility:g}')
    ax.set(xlim=(0, wealth*1.1), ylim=(0, (1+r)*wealth*1.1),
           xlabel='First-period consumption $c_0$', ylabel='Second-period consumption $c_1$',
           title='Indifference curves: log utility')
    ax.legend()
    return ax


def SolveUtilityMaximizationProblem(y_0, y_1, beta, r):
    if not 0 < beta < 1:
        raise ValueError('Use 0 < beta < 1.')
    wealth = _budget(y_0, y_1, r)
    # Derive optimal choices from the Lagrangian FOCs, then plot them.
    solution = SolveUtilityMaximizationLagrange(y_0, y_1, beta, r, show_steps=False)
    c0, c1 = solution['c_0'], solution['c_1']
    saving, utility = solution['a'], solution['utility']
    ax = PlotBudgetSet(y_0, y_1, r)
    x = np.linspace(wealth/1000, wealth*1.1, 600)
    ax.plot(x, np.exp((utility-np.log(x))/beta), label='Optimal indifference curve')
    ax.plot(x, beta*(1+r)*x, ls='--', label='Euler equation')
    ax.scatter([c0], [c1], color='black', zorder=4, label='Optimal consumption')
    ax.set_title('Optimal two-period consumption and saving')
    ax.legend()
    print(f'c_0 = {c0:.6f}, c_1 = {c1:.6f}, saving = {saving:.6f}, utility = {utility:.6f}')
    assert np.isclose(c0+c1/(1+r), wealth)
    return {'c_0': c0, 'c_1': c1, 'a': saving, 'utility': utility}

def SolveUtilityMaximizationLagrange(y_0, y_1, beta, r, show_steps=True):
    """Derive and solve the two-period log-utility problem symbolically.

    Borrowing is allowed at the saving rate. The binding present-value budget
    follows from positive marginal utilities. Returns symbolic derivations,
    optimal numerical choices, and checks of the first-order/KKT conditions.
    SymPy is imported only when this function is called.
    """
    import sympy as sp

    values = np.asarray([y_0, y_1, beta, r], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('All inputs must be finite real numbers.')
    y_0, y_1, beta, r = map(float, values)
    if not 0 < beta < 1:
        raise ValueError('Use 0 < beta < 1.')
    wealth = _budget(y_0, y_1, r)

    c0, c1 = sp.symbols('c_0 c_1', positive=True)
    multiplier = sp.Symbol('lambda', positive=True)
    income0, income1, interest = sp.symbols('y_0 y_1 r', real=True)
    discount = sp.Symbol('beta', positive=True)
    utility = sp.log(c0) + discount*sp.log(c1)
    budget_slack = income0 + income1/(1+interest) - c0 - c1/(1+interest)
    lagrangian = utility + multiplier*budget_slack

    # Actually differentiate and solve: numerical choices are not hard-coded.
    focs = [sp.Eq(sp.diff(lagrangian, variable), 0)
            for variable in [c0, c1, multiplier]]
    candidates = sp.solve(focs, [c0, c1, multiplier], dict=True)
    if len(candidates) != 1:
        raise RuntimeError(f'Expected a unique symbolic solution, got {len(candidates)}.')
    optimum = {key: sp.factor(value) for key, value in candidates[0].items()}
    saving_expression = sp.factor(income0-optimum[c0])
    substitutions = {income0: y_0, income1: y_1, discount: beta, interest: r}
    consumption0, consumption1, shadow_price = [float(optimum[v].subs(substitutions))
                                               for v in [c0, c1, multiplier]]
    if min(consumption0, consumption1, shadow_price) <= 0:
        raise RuntimeError('The candidate is outside the positive interior domain.')
    saving = float(saving_expression.subs(substitutions))
    max_utility = float(utility.subs({c0: consumption0, c1: consumption1, discount: beta}))
    residuals = [float((eq.lhs-eq.rhs).subs({**substitutions, c0: consumption0,
                                           c1: consumption1, multiplier: shadow_price}))
                 for eq in focs]
    budget_error = consumption0+consumption1/(1+r)-wealth
    if not np.allclose(residuals, 0, atol=1e-9, rtol=0):
        raise RuntimeError('First-order conditions were not satisfied numerically.')
    if not np.isclose(budget_error, 0, atol=1e-9, rtol=1e-10):
        raise RuntimeError('The solution does not satisfy the budget constraint.')

    if show_steps:
        from IPython.display import display, Markdown, Math
        display(Markdown('### 1. Objective and budget constraint'))
        display(Math(r'\max_{c_0,c_1>0}\;' + sp.latex(utility)))
        display(Math(sp.latex(sp.Ge(budget_slack, 0))))
        display(Markdown('Utility increases in both consumptions, so the budget binds at the optimum.'))
        display(Markdown('### 2. Construct the Lagrangian'))
        display(Math(r'\mathcal{L}=' + sp.latex(lagrangian)))
        display(Markdown('### 3. Differentiate to obtain the first-order conditions'))
        for variable, eq in zip([c0, c1, multiplier], focs):
            display(Math(r'\frac{\partial\mathcal{L}}{\partial '+sp.latex(variable)+'}=' + sp.latex(eq)))
        display(Markdown('### 4. Solve the equations simultaneously'))
        for variable in [c0, c1, multiplier]:
            display(Math(sp.latex(variable)+'^{*}='+sp.latex(optimum[variable])))
        display(Math(r'a^{*}='+sp.latex(saving_expression)))
        display(Markdown('The utility Hessian is negative definite for positive consumption and beta. With an affine budget set, these conditions identify the unique global maximum. The positive multiplier and binding budget also satisfy complementary slackness.'))
        display(Math(r'\nabla^2 U=' + sp.latex(sp.hessian(utility, [c0, c1]))))
        display(Markdown('### 5. Substitute the parameter values'))
        print(f'y_0={y_0:g}, y_1={y_1:g}, beta={beta:g}, r={r:g}')
        print(f'Optimal c_0 = {consumption0:.6f}')
        print(f'Optimal c_1 = {consumption1:.6f}')
        print(f'Optimal saving a = {saving:.6f}')
        print(f'Lagrange multiplier = {shadow_price:.6f}')
        print(f'Maximum utility = {max_utility:.6f}')
        print(f'Maximum FOC residual = {max(abs(v) for v in residuals):.2e}')
        print(f'Budget residual = {budget_error:.2e}')

    return {'c_0': consumption0, 'c_1': consumption1, 'a': saving,
            'lambda': shadow_price, 'utility': max_utility,
            'lagrangian': lagrangian, 'focs': focs, 'symbolic_solution': optimum,
            'symbolic_saving': saving_expression, 'foc_residuals': residuals,
            'budget_residual': budget_error}


def InterpretLagrangeMultiplier(solution, r, delta_wealth=0.01):
    """Display the numerical shadow-value example for a two-period log-utility solution."""
    example_c0 = solution['c_0']
    example_c1 = solution['c_1']
    example_lambda = solution['lambda']
    example_r = float(r)
    # Recover the discount factor used in that solution from its Euler equation.
    example_beta = example_c1 / ((1+example_r)*example_c0)
    example_wealth = example_c0 + example_c1/(1+example_r)
    approx_utility_change = example_lambda * delta_wealth
    # For log utility, both optimal consumptions scale proportionally with wealth.
    exact_utility_change = (1+example_beta) * np.log1p(delta_wealth/example_wealth)
    print(f'Optimal current consumption: {example_c0:.6f}')
    print(f'Lambda = 1/c_0: {example_lambda:.6f}')
    print(f'Present-value wealth: {example_wealth:.6f}')
    print(f'Wealth increase: {delta_wealth:.2f}')
    print(f'Approximate increase in maximum utility: {approx_utility_change:.8f}')
    print(f'Exact increase in maximum utility:       {exact_utility_change:.8f}')
    print(f'Marginal utility value of one extra unit of future income: {example_lambda/(1+example_r):.6f}')
    assert np.isclose(example_lambda, 1/example_c0)
    assert np.isclose(example_lambda, example_beta*(1+example_r)/example_c1)
