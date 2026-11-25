import numpy as np
import pytest
import torch
from scipy.stats import ks_2samp

from iiot_qelm.models.encoder import AttentionEncoder, EncoderConfig, Phase1Network
from iiot_qelm.models.focal import LossConfig, cross_entropy_from_logits, focal_loss_from_logits
from iiot_qelm.models.qelm import QELM, QELMConfig, activate, quantum_weights, theta_inertness


# ---------------------------------------------------------------- encoder shapes
@pytest.mark.parametrize("b", [1, 2, 33])
def test_encoder_shapes_for_batch_sizes(b):
    enc = AttentionEncoder().eval()
    assert enc(torch.rand(b, 48)).shape == (b, 256)


def test_token_expansion_is_the_documented_reconstruction():
    c = EncoderConfig()
    enc = AttentionEncoder(c)
    assert (c.bottleneck_dim, c.num_tokens, c.token_dim) == (64, 8, 32)
    assert enc.token_proj.in_features == 64 and enc.token_proj.out_features == 8 * 32       # 64 -> 256, reshaped to (8, 32)
    assert enc.attn.q.out_features == enc.attn.k.out_features == enc.attn.v.out_features == 8 * 32
    assert enc.attn.o.in_features == 256 and enc.attn.o.out_features == 32
    assert enc.attn.q.weight is not enc.attn.k.weight and not torch.equal(enc.attn.q.weight, enc.attn.k.weight)    # distinct Q, K, V
    assert enc.ffn[0].out_features == 4 * 32 and c.dropout == 0.1 and c.out_dim == 256


def test_feature_index_embedding_is_learned_and_fixed_at_inference():
    enc = AttentionEncoder()
    assert isinstance(enc.index_embedding, torch.nn.Parameter) and enc.index_embedding.shape == (8, 32)
    assert "index_embedding" in enc.state_dict()
    enc.eval()
    x = torch.rand(4, 48)
    assert torch.equal(enc(x), enc(x))                         # deterministic in eval mode
    before = enc.index_embedding.clone()
    enc(x)
    assert torch.equal(before, enc.index_embedding)            # inference never changes it
    with torch.no_grad():
        enc.index_embedding += 1.0
    assert not torch.allclose(enc(x), AttentionEncoder().eval()(x))


def test_encoder_rejects_wrong_input_width():
    with pytest.raises(ValueError):
        AttentionEncoder()(torch.rand(2, 47))


def test_phase1_heads():
    assert Phase1Network(15, head="qelm")(torch.rand(3, 48)).shape == (3, 15)
    mlp = Phase1Network(2, head="mlp")
    assert mlp(torch.rand(3, 48)).shape == (3, 2) and sum(isinstance(m, torch.nn.Linear) for m in mlp.head) == 4   # 3 hidden + output
    with pytest.raises(ValueError):
        Phase1Network(2, head="nope")


# ---------------------------------------------------------------- focal loss
def test_focal_gamma0_alpha1_equals_weighted_cross_entropy():
    torch.manual_seed(0)
    lg, y = torch.randn(64, 5), torch.randint(0, 5, (64,))
    w = torch.rand(5) + 0.5
    assert torch.allclose(focal_loss_from_logits(lg, y, gamma=0.0, alpha=1.0), cross_entropy_from_logits(lg, y), atol=1e-6)
    assert torch.allclose(focal_loss_from_logits(lg, y, gamma=0.0, alpha=1.0, class_weights=w), cross_entropy_from_logits(lg, y, w), atol=1e-6)


def test_focal_is_finite_and_has_finite_gradients_at_extreme_logits():
    lg = torch.tensor([[1e4, -1e4, 0.0], [-1e4, 1e4, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    loss = focal_loss_from_logits(lg, torch.tensor([0, 0, 2]), gamma=2.0, alpha=0.25)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(lg.grad).all()


def test_focal_downweights_easy_examples():
    lg = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    easy = focal_loss_from_logits(lg, torch.tensor([0, 1]), gamma=2.0, alpha=1.0, reduction="none")
    hard = focal_loss_from_logits(lg, torch.tensor([1, 0]), gamma=2.0, alpha=1.0, reduction="none")
    assert (hard > 100 * easy).all()


def test_class_weights_enter_exactly_once():
    lg, y = torch.randn(32, 3), torch.randint(0, 3, (32,))
    w = torch.tensor([1.0, 2.0, 3.0])
    plain = focal_loss_from_logits(lg, y, reduction="none")
    weighted = focal_loss_from_logits(lg, y, class_weights=w, reduction="none")
    assert torch.allclose(weighted, plain * w[y])
    cfg = LossConfig(kind="focal", use_class_weights=False)
    assert torch.allclose(cfg(lg, y, w), focal_loss_from_logits(lg, y))          # config can switch weighting off


def test_alpha_semantics():
    lg, y = torch.randn(10, 2), torch.randint(0, 2, (10,))
    uni = focal_loss_from_logits(lg, y, alpha=0.25, reduction="none")
    bin_ = focal_loss_from_logits(lg, y, alpha=0.25, alpha_mode="binary", reduction="none")
    assert torch.allclose(bin_[y == 1], uni[y == 1]) and torch.allclose(bin_[y == 0], uni[y == 0] * 3.0)      # 1 - alpha = 3 alpha for alpha = .25
    pc = focal_loss_from_logits(lg, y, alpha=[0.1, 0.9], alpha_mode="per_class", reduction="none")
    assert torch.allclose(pc, focal_loss_from_logits(lg, y, alpha=1.0, reduction="none") * torch.tensor([0.1, 0.9])[y])
    with pytest.raises(ValueError):
        focal_loss_from_logits(torch.randn(4, 3), torch.randint(0, 3, (4,)), alpha_mode="binary")
    with pytest.raises(ValueError):
        focal_loss_from_logits(lg, y, alpha_mode="nope")
    with pytest.raises(ValueError):
        focal_loss_from_logits(torch.randn(4), y[:4])
    assert focal_loss_from_logits(lg, y, reduction="sum") > 0


def test_loss_config_dispatch():
    lg, y = torch.randn(8, 3), torch.randint(0, 3, (8,))
    assert torch.allclose(LossConfig(kind="ce")(lg, y), cross_entropy_from_logits(lg, y))
    with pytest.raises(ValueError):
        LossConfig(kind="hinge")(lg, y)


# ---------------------------------------------------------------- QELM
def _data(n=300, d=12, k=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    return rng.normal(size=(n, d)) + 2.0 * np.eye(k, d)[y], y


def test_qelm_seeded_determinism_and_seed_sensitivity():
    X, y = _data()
    a = QELM(QELMConfig(n_layers=2, width=64, seed=5)).fit(X, y)
    b = QELM(QELMConfig(n_layers=2, width=64, seed=5)).fit(X, y)
    c = QELM(QELMConfig(n_layers=2, width=64, seed=6)).fit(X, y)
    assert np.array_equal(a.beta, b.beta) and not np.allclose(a.beta, c.beta)


def test_rho_endpoints_give_cosine_and_sine_projections():
    n = 5000
    for rho, fn in ((1.0, np.cos), (0.0, np.sin)):
        rng = np.random.default_rng(0)
        phi = np.random.default_rng(0).uniform(0, 2 * np.pi, (n, 1))
        w = quantum_weights(rng, n, 1, rho, 0.0, "phase_shift", scale=False)
        assert np.allclose(w, fn(phi))


def test_theta_changes_draws_but_not_their_distribution():
    rep = theta_inertness(rho=0.63, thetas=(0.0, 1.84))
    assert abs(rep[0.0]["std"] - rep[1.84]["std"]) < 0.01 and abs(rep[0.0]["mean"] - rep[1.84]["mean"]) < 0.01
    r1, r2 = np.random.default_rng(1), np.random.default_rng(1)
    w0 = quantum_weights(r1, 2000, 1, 0.63, 0.0).ravel()
    w1 = quantum_weights(r2, 2000, 1, 0.63, 1.84).ravel()
    assert not np.allclose(w0, w1) and ks_2samp(w0, w1).pvalue > 0.001          # different values, same distribution => theta is inert
    assert np.allclose(quantum_weights(np.random.default_rng(1), 50, 3, 0.63, 1.84, "none"), quantum_weights(np.random.default_rng(1), 50, 3, 0.63, 0.0, "none"))


def test_closed_form_matches_direct_ridge_solution():
    X, y = _data()
    q = QELM(QELMConfig(n_layers=1, width=32, lambda1=1e-3, lambda2=2e-3, seed=0, solve_batch=64)).fit(X, y)
    H = q.hidden(X)
    T = np.eye(3)[y]
    direct = np.linalg.solve(H.T @ H + 3e-3 * np.eye(32), H.T @ T)
    assert np.allclose(q.beta, direct, atol=1e-8) and q.solver_info["method"] == "cholesky"
    q2 = QELM(QELMConfig(n_layers=1, width=32, lambda1=1e-3, lambda2=2e-3, seed=0, solve_batch=10_000)).fit(X, y)
    assert np.allclose(q.beta, q2.beta)                          # batch accumulation is exact


def test_both_lambdas_are_ridge_terms_only_their_sum_matters():
    X, y = _data()
    a = QELM(QELMConfig(n_layers=1, width=16, lambda1=1e-3, lambda2=1e-3, seed=0)).fit(X, y)
    b = QELM(QELMConfig(n_layers=1, width=16, lambda1=2e-3, lambda2=0.0, seed=0)).fit(X, y)
    assert np.allclose(a.beta, b.beta)


def test_solver_survives_ill_conditioning():
    X, y = _data()
    q = QELM(QELMConfig(n_layers=1, width=16, lambda1=0.0, lambda2=0.0, seed=0))
    q._build(X.shape[1])
    A = np.ones((16, 16))                                          # rank one, singular without a penalty
    B = np.ones((16, 3))
    beta = q._solve(A, B)
    assert np.isfinite(beta).all() and q.solver_info["jitter"] > 0 or q.solver_info["method"] == "lstsq"
    q.solver_info = {}
    q.fit(np.zeros((20, 5)), np.arange(20) % 3)                   # all-zero inputs => degenerate hidden layer
    assert np.isfinite(q.beta).all()


def test_qelm_activations_and_validation():
    x = np.linspace(-3, 3, 7)
    assert np.allclose(activate("relu", x), np.maximum(x, 0)) and np.allclose(activate("tanh", x), np.tanh(x))
    assert np.allclose(activate("sigmoid", np.array([-1e3, 0, 1e3])), [0, 0.5, 1]) and abs(activate("gelu", np.array([0.0]))[0]) < 1e-12
    with pytest.raises(ValueError):
        activate("swish", x)
    for bad in (dict(rho=1.5), dict(activation="x"), dict(n_layers=0), dict(lambda1=-1.0), dict(init="nope")):
        with pytest.raises(ValueError):
            QELM(QELMConfig(**bad))


def test_qelm_not_fitted_errors_and_save_load_parity(tmp_path):
    X, y = _data()
    q = QELM(QELMConfig(n_layers=2, width=32))
    with pytest.raises(RuntimeError):
        q.decision_scores(X)
    with pytest.raises(RuntimeError):
        q.hidden(X)
    q.fit(X, y)
    q.save(tmp_path / "q.npz")
    r = QELM.load(tmp_path / "q.npz")
    assert np.array_equal(q.decision_scores(X), r.decision_scores(X)) and r.cfg == q.cfg and q.n_params()["solved"] == 32 * 3
    p = q.predict_proba(X)
    assert np.allclose(p.sum(1), 1) and (q.predict(X) == p.argmax(1)).all()
    with pytest.raises(ValueError):
        QELM().fit_batches(iter([]), 3, 4)


def test_classical_init_differs_from_quantum_init():
    X, y = _data()
    a = QELM(QELMConfig(n_layers=1, width=32, init="quantum", seed=0)).fit(X, y)
    b = QELM(QELMConfig(n_layers=1, width=32, init="classical", seed=0)).fit(X, y)
    assert not np.allclose(a.W[0], b.W[0])
    assert QELM(QELMConfig(n_layers=1, width=8, init="classical", weight_scale="none")).fit(X, y).W[0].std() > 0.5
