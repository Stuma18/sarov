(defrule MAIN::CO2
   (C)
   (O2)
   =>
   (assert (CO2)))

(defrule MAIN::H2O
   (H2)
   (MgO)
   =>
   (assert (Mg))
   (assert (H2O)))

(defrule MAIN::H2CO3
   (H2O)
   (CO2)
   =>
   (assert (H2CO3))
   (printout t "H2CO3 was synthesized" crlf))

