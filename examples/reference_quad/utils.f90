LOGICAL FUNCTION LSAME( CA, CB )
  CHARACTER CA, CB
  CHARACTER(1) UCA, UCB
  UCA = CA
  UCB = CB
  IF( UCA >= 'a' .AND. UCA <= 'z' ) UCA = CHAR( ICHAR( UCA ) - 32 )
  IF( UCB >= 'a' .AND. UCB <= 'z' ) UCB = CHAR( ICHAR( UCB ) - 32 )
  LSAME = UCA == UCB
END FUNCTION

SUBROUTINE XERBLA( SRNAME, INFO )
  CHARACTER(*) SRNAME
  INTEGER INFO
  WRITE(*,*) ' ** On entry to ', SRNAME, ' parameter number ', INFO, ' had an illegal value'
END SUBROUTINE
