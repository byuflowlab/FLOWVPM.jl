
# the original function is an n-body interaction. passing reversediff straight through slows down differentation by two orders of magnitude (at least)
function Estr_direct_singlethreaded(pfield::ParticleField{R, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, useGPU}) where {R<:ReverseDiff.TrackedReal, F<:Formulation, V<:ViscousScheme, TUinf, S<:SubFilterScale, Tkernel, TUJ, Tintegration, TR, useGPU}
    
    tp = ReverseDiff.tape(pfield)
    T = ReverseDiff.valtype(R)
    tx = zeros(T, 3)
    dx = zeros(T, 3)
    N = zeros(T, 3)
    for target_particle in iterator(pfield)
        for a=1:3
            tx[a] = target_particle[a].value
        end
        for source_particle in iterator(pfield)
            r2 = zero(T)
            for a=1:3
                dx[a] = source_particle[a].value - tx[a]
                r2 += dx[a]*dx[a]
            end
            r = sqrt(r2)

            ΓS = get_Gamma(source_particle)
            JS = get_J(source_particle)
            JT = get_J(target_particle)
            σ = get_sigma(source_particle)[]
            for a=1:3
                N[a] = zero(T)
                for b=1:3
                    # N[a] = (JS[a, b] - JT[a, b])*ΓS[b]
                    # J[a,b] -> J[a + 3*(b-1)]
                    if pfield.transposed
                        N[a] += (JT[b + 3*(a-1)].value - JS[b + 3*(a-1)].value)*ΓS[b].value
                    else
                        N[a] += (JT[a + 3*(b-1)].value - JS[a + 3*(b-1)].value)*ΓS[b].value
                    end
                end
            end

            ζ_σ = pfield.kernel.zeta(r/σ.value) * σ.value^-3

            for a=1:3
                target_particle[SFS_INDEX[a]].value += ζ_σ*N[a]
            end
        end
    end

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        Estr_direct_singlethreaded,
                        pfield,
                        nothing)
    return nothing

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(Estr_direct_singlethreaded)})

    pfield = instruction.input
    
    T = ReverseDiff.valtype(eltype(pfield))
    tx = zeros(T, 3)
    dx = zeros(T, 3)
    N = zeros(T, 3)
    Nbar = zeros(T, 3)
    #@show ReverseDiff.deriv.(pfield.particles[X_INDEX, :])
    #dζ_f(_x) = ForwardDiff.derivative(pfield.kernel.zeta, _x)
    for target_particle in iterator(pfield)
        for a=1:3
            tx[a] = target_particle[a].value
        end
        for source_particle in iterator(pfield)
            r2 = zero(T)
            for a=1:3
                dx[a] = source_particle[a].value - tx[a]
                r2 += dx[a]*dx[a]
            end
            r = sqrt(r2)

            ΓS = get_Gamma(source_particle)
            JS = get_J(source_particle)
            JT = get_J(target_particle)
            σ = get_sigma(source_particle)[]
            σv = σ.value

            for a=1:3
                N[a] = zero(T)
                for b=1:3
                    if pfield.transposed
                        N[a] += (JT[b + 3*(a-1)].value - JS[b + 3*(a-1)].value)*ΓS[b].value
                    else
                        N[a] += (JT[a + 3*(b-1)].value - JS[a + 3*(b-1)].value)*ΓS[b].value
                    end
                end
            end

            ζ = pfield.kernel.zeta(r/σv)
            dζ = ForwardDiff.derivative(pfield.kernel.zeta, r/σv)
            #dζ = dζ_f(r/σv)
            ζ_σ = ζ * σv^-3

            # Running the reverse pass means reverting the change from the forward pass.
            # This is cheaper than storing the old version,
            # since we have to recompute the intermediate terms for the pullback anyway.
            for a=1:3
                target_particle[SFS_INDEX[a]].value -= ζ_σ*N[a]
            end

            ζ_σ_bar = zero(T)
            # ζ_σ_bar = Sbar_a*Na
            # Nbar_a = Sbar_a*ζ_σ
            for a=1:3
                ζ_σ_bar += target_particle[SFS_INDEX[a]].deriv*N[a]
                Nbar[a] = target_particle[SFS_INDEX[a]].deriv*ζ_σ
            end
            # if r == 0, then a bunch of terms cancel mathematically with a removeable singularity. For better numerical conditioning we handle this case separately.
            if r > 0
                rbar = dζ*σv^(-4)*ζ_σ_bar

                for a=1:3
                    dxbar_a = rbar*dx[a]/r
                    ReverseDiff._add_to_deriv!(source_particle[a], dxbar_a)
                    ReverseDiff._add_to_deriv!(target_particle[a], -dxbar_a)
                end
                # σ_bar += -σ^-4 * ζ_σ_bar * [dζ_dσ * r * σ^-1 + 3*ζ]
                ReverseDiff._add_to_deriv!(source_particle[SIGMA_INDEX], -σv^(-4)*ζ_σ_bar*(dζ*r*σv^-1 + 3*ζ))
            else
                # r = 0 -> dζ_dσ = 0 (assuming ζ is smooth and even). also dx -> 0 and so rbar -> 0. following this through, dxbar = 0.
                # σ_bar += -σ^-4 * ζ_σ_bar * [dζ_dσ * r * σ^-1 + 3*ζ]
                ReverseDiff._add_to_deriv!(source_particle[SIGMA_INDEX], -σv^(-4)*ζ_σ_bar*3*ζ)
            end

            for a=1:3
                for b=1:3
                    if pfield.transposed
                        ReverseDiff._add_to_deriv!(JT[b + 3*(a-1)], Nbar[a]*ΓS[b].value)
                        ReverseDiff._add_to_deriv!(JS[b + 3*(a-1)], -Nbar[a]*ΓS[b].value)
                        ReverseDiff._add_to_deriv!(ΓS[b], (JT[b + 3*(a-1)].value - JS[b + 3*(a-1)].value)*Nbar[a])
                    else
                        ReverseDiff._add_to_deriv!(JT[a + 3*(b-1)], Nbar[a]*ΓS[b])
                        ReverseDiff._add_to_deriv!(JS[a + 3*(b-1)], -Nbar[a]*ΓS[b])
                        ReverseDiff._add_to_deriv!(ΓS[b], (JT[a + 3*(b-1)].value - JS[a + 3*(b-1)].value)*Nbar[a])
                    end
                end
            end

        end
    end
    #@show ReverseDiff.deriv.(pfield.particles[X_INDEX, :])
    # Sbar is not unseeded because it is accumulated into.
    return nothing

end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(Estr_direct_singlethreaded)})

    pfield = instruction.input
    T = ReverseDiff.valtype(eltype(pfield))
    tx = zeros(T, 3)
    dx = zeros(T, 3)
    N = zeros(T, 3)
    for target_particle in iterator(pfield)
        for a=1:3
            tx[a] = target_particle[a].value
        end
        for source_particle in iterator(pfield)
            r2 = zero(T)
            for a=1:3
                dx[a] = source_particle[a].value - tx[a]
                r2 += dx[a]*dx[a]
            end
            r = sqrt(r2)

            ΓS = get_Gamma(source_particle)
            JS = get_J(source_particle)
            JT = get_J(target_particle)
            σ = get_sigma(source_particle)[]

            for a=1:3
                N[a] = zero(T)
                for b=1:3
                    if pfield.transposed
                        N[a] = (JT[b + 3*(a-1)].value - JS[b + 3*(a-1)].value)*ΓS[b].value
                    else
                        N[a] = (JT[a + 3*(b-1)].value - JS[a + 3*(b-1)].value)*ΓS[b].value
                    end
                end
            end

            ζ_σ = pfield.kernel.zeta(r/σ.value) * σ.value^-3

            for a=1:3
                target_particle[SFS_INDEX[a]].value += ζ_σ*N[a]
            end
        end
    end
    return nothing

end

# original unrolled function
    #=for target_particle in iterator(pfield)
        #is_static(target_particle) && continue
        tx, ty, tz = target_particle[1], target_particle[2], target_particle[3]
        for source_particle in iterator(pfield)
            sx, sy, sz = source_particle[1], source_particle[2], source_particle[3]

            dx, dy, dz = sx - tx, sy - ty, sz - tz
            r2 = dx * dx + dy * dy + dz * dz
            r = r2 > 0 ? sqrt(r2) : zero(typeof(r2)) # AD derivatives are corrupted with NaNs without this check.
            #r = sqrt(dx * dx + dy * dy + dz * dz)

            #Estr_direct(target_particle, source_particle, r, pfield.kernel.zeta, pfield.transposed)
            GS = get_Gamma(source_particle)
            JS = get_J(source_particle)
            JT = get_J(target_particle)

            # Stretching term
            if transposed
                # Transposed scheme (Γq⋅∇')(Up - Uq)
                S1 = (JT[1] - JS[1])*GS[1]+(JT[2] - JS[2])*GS[2]+(JT[3] - JS[3])*GS[3]
                S2 = (JT[4] - JS[4])*GS[1]+(JT[5] - JS[5])*GS[2]+(JT[6] - JS[6])*GS[3]
                S3 = (JT[7] - JS[7])*GS[1]+(JT[8] - JS[8])*GS[2]+(JT[9] - JS[9])*GS[3]
            else
                # Classic scheme (Γq⋅∇)(Up - Uq)
                S1 = (JT[1] - JS[1])*GS[1]+(JT[4] - JS[4])*GS[2]+(JT[7] - JS[7])*GS[3]
                S2 = (JT[2] - JS[2])*GS[1]+(JT[5] - JS[5])*GS[2]+(JT[8] - JS[8])*GS[3]
                S3 = (JT[3] - JS[3])*GS[1]+(JT[6] - JS[6])*GS[2]+(JT[9] - JS[9])*GS[3]
            end

            sigma_inv = 1.0 / get_sigma(source_particle)[]
            zeta_sgm = zeta(r*sigma_inv) * sigma_inv * sigma_inv * sigma_inv

            # Add ζ_σ (Γq⋅∇)(Up - Uq)
            get_SFS(target_particle)[1] += zeta_sgm*S1
            get_SFS(target_particle)[2] += zeta_sgm*S2
            get_SFS(target_particle)[3] += zeta_sgm*S3
        end
    end=#
