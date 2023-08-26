import paddle


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.astype(paddle.float32)
            gl = gl.astype(paddle.float32)
            loss += paddle.mean(paddle.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.astype(paddle.float32)
        dg = dg.astype(paddle.float32)
        r_loss = paddle.mean((1 - dr) ** 2)
        g_loss = paddle.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.astype(paddle.float32)
        l = paddle.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.astype(paddle.float32)
    logs_q = logs_q.astype(paddle.float32)
    m_p = m_p.astype(paddle.float32)
    logs_p = logs_p.astype(paddle.float32)
    z_mask = z_mask.astype(paddle.float32)

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * paddle.exp(-2. * logs_p)
    kl = paddle.sum(kl * z_mask)
    l = kl / paddle.sum(z_mask)
    return l
