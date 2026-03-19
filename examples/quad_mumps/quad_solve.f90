! quad_solve.f90 — sparse solver in quad precision (REAL(16) = IEEE binary128).
!
! Exports:
!   int sparse_solve(int n, int nnz, int *irn, int *jcn,
!                    void *a, void *rhs, int sym)
!
! Converts sparse COO to dense, solves via LU factorisation with partial
! pivoting in quad precision.
!
! Build:
!   gfortran -O2 -shared -fPIC -o quad_solve.so quad_solve.f90

INTEGER(C_INT) FUNCTION SPARSE_SOLVE(N, NNZ, IRN, JCN, A, RHS, SYM) &
    BIND(C, NAME='sparse_solve')
  USE, INTRINSIC :: ISO_C_BINDING
  IMPLICIT NONE

  INTEGER(C_INT), VALUE, INTENT(IN) :: N, NNZ, SYM
  INTEGER(C_INT), INTENT(INOUT)     :: IRN(NNZ), JCN(NNZ)
  REAL(16), INTENT(INOUT)           :: A(NNZ), RHS(N)

  REAL(16), ALLOCATABLE :: DENSE(:,:), B(:)
  INTEGER, ALLOCATABLE  :: IPIV(:)
  INTEGER :: K, I, J, JJ, IMAX
  REAL(16) :: MAXVAL_ABS, TEMP, FACTOR

  SPARSE_SOLVE = 0

  ! --- Build dense matrix from COO ---
  ALLOCATE(DENSE(N, N), B(N), IPIV(N))
  DENSE = 0.0Q0

  DO K = 1, NNZ
    I = IRN(K)
    J = JCN(K)
    DENSE(I, J) = DENSE(I, J) + A(K)
    IF (SYM /= 0 .AND. I /= J) THEN
      DENSE(J, I) = DENSE(J, I) + A(K)
    END IF
  END DO

  ! Copy RHS
  B(1:N) = RHS(1:N)

  ! --- LU factorisation with partial pivoting ---
  DO K = 1, N
    ! Find pivot
    IMAX = K
    MAXVAL_ABS = ABS(DENSE(K, K))
    DO I = K + 1, N
      IF (ABS(DENSE(I, K)) > MAXVAL_ABS) THEN
        MAXVAL_ABS = ABS(DENSE(I, K))
        IMAX = I
      END IF
    END DO
    IPIV(K) = IMAX

    IF (MAXVAL_ABS == 0.0Q0) THEN
      SPARSE_SOLVE = K   ! singular
      DEALLOCATE(DENSE, B, IPIV)
      RETURN
    END IF

    ! Swap rows K and IMAX
    IF (IMAX /= K) THEN
      DO JJ = 1, N
        TEMP = DENSE(K, JJ)
        DENSE(K, JJ) = DENSE(IMAX, JJ)
        DENSE(IMAX, JJ) = TEMP
      END DO
      TEMP = B(K)
      B(K) = B(IMAX)
      B(IMAX) = TEMP
    END IF

    ! Eliminate below
    DO I = K + 1, N
      FACTOR = DENSE(I, K) / DENSE(K, K)
      DENSE(I, K) = FACTOR   ! store L factor
      DO JJ = K + 1, N
        DENSE(I, JJ) = DENSE(I, JJ) - FACTOR * DENSE(K, JJ)
      END DO
      B(I) = B(I) - FACTOR * B(K)
    END DO
  END DO

  ! --- Back substitution ---
  DO I = N, 1, -1
    DO JJ = I + 1, N
      B(I) = B(I) - DENSE(I, JJ) * B(JJ)
    END DO
    B(I) = B(I) / DENSE(I, I)
  END DO

  ! Write solution back to RHS
  RHS(1:N) = B(1:N)

  DEALLOCATE(DENSE, B, IPIV)
END FUNCTION
