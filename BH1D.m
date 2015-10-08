% 
% Solves the T=0 Bose-Hubbard chain
% (c) Vincenzo Savona
% 

clear all

% Define parameters (energies in units of hopping energy T)
Nmax = 8;                       % Max number of quanta per site
Nsites = 8;                     % Number of sites
Np = Nsites;                    % Number of particles
Nstates = (Nmax + 1)^Nsites;    % Total number of Fock states
U = 10.0;                       % Bose-Hubbard energy

% Define creation and destruction operators on single-site Hilbert space
A = spdiags(sqrt(0:Nmax)',1,Nmax+1,Nmax+1); % Destruction operator on single-mode space
Ac = A.';                                   % Creation operator on single-mode space

% Define operators in the N-site Hilbert space as Kroeneker tensor
% products of operators in the single-site space

An = cell(1,Nsites);            % Destruction operator
Anc = cell(1,Nsites);           % Destruction operator
Itot = speye(Nmax+1);           % Site-incremental identity operator
Ntot = Ac * A;                  % Site-incremental total number operator
An{1} = A;                      % One-site chain: destruction operator
Anc{1} = Ac;                    % One-site chain: creation operator

% Build chain operator incrementally site by site. After each site, retain
% only basis states with Ntot<=Nmax. In this way the an excessive growth of
% the Hilbert space size is avoided at all stages during the building.
for j=2:Nsites
    An{j} = kron(Itot,A);
    Anc{j} = An{j}.';
    for k=1:j-1
        An{k} = kron(An{k},speye(Nmax+1));
        Anc{k} = kron(Anc{k},speye(Nmax+1));
    end
    Ntot = kron(Ntot,speye(Nmax+1)) + kron(Itot,Ac*A);
    Itot = kron(Itot,speye(Nmax+1));
    Nt = round(full(diag(Ntot)));            % Number of particles in each state
    jmax = find(Nt <= Nmax);          % Find states withno more than Np particles
    Ntot = Ntot(jmax,jmax);
    Itot = Itot(jmax,jmax);
    for k=1:j
        ttt = An{k};
        ttt = ttt(jmax,jmax);
        An{k} = ttt;
        ttt = Anc{k};
        ttt = ttt(jmax,jmax);
        Anc{k} = ttt;
    end
end
Nstates = length(jmax);         % New total number of states

Nn = cell(1,Nsites);            % Number operator
for j=1:Nsites
    Nn{j} = Anc{j} * An{j};
end

% Builds an array with sites occupation numbers for each quantum state to
% be used for computing importance sampling
states = zeros(Nsites,Nstates);
for j=1:Nstates
    for k=1:Nsites
        states(k,j) = round(Nn{k}(j,j));
    end
end

% Define the interaction part of the Hamiltonian
H = sparse(Nstates,Nstates);
for j=1:Nsites
    H = H + U/2 .* (Nn{j} * (Nn{j} - speye(Nstates)));
end
% Add the kinetic contribution
for j=2:Nsites
    H = H - Anc{j} * An{j-1} - Anc{j-1} * An{j};
end
H = H - Anc{1} * An{Nsites} - Anc{Nsites} * An{1};

% Select states with N particles only
jmax = find(round(full(diag(Ntot))) == Np);          % Find states with Np particles
Nstates = length(jmax);         % New total number of states

states = states(:,jmax);
H = H(jmax,jmax);               % Project Hamiltonian
% Project other previously defined operators
for j=1:Nsites
    ttt = An{j};
    ttt = ttt(jmax,jmax);
    An{j} = ttt;
    ttt = Anc{j};
    ttt = ttt(jmax,jmax);
    Anc{j} = ttt;
    ttt = Nn{j};
    ttt = ttt(jmax,jmax);
    Nn{j} = ttt;
end

[~,a] = eigs(H,1,-4)



