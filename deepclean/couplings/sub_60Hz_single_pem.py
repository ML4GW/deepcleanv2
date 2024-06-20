from deepclean.couplings import Coupling, SubtractionProblem


class Sub60Hz_single_pem(SubtractionProblem):
    description = "Subtraction of 60Hz main noise"

    H1 = Coupling(
        55,
        65,
        [
            "PEM-CS_MAINSMON_EBAY_1_DQ",
            # "ASC-INP1_P_IN1_DQ",
            # "ASC-INP1_Y_IN1_DQ",
            # "ASC-MICH_P_IN1_DQ",
            # "ASC-MICH_Y_IN1_DQ",
            # "ASC-PRC1_P_IN1_DQ",
            # "ASC-PRC1_Y_IN1_DQ",
            # "ASC-PRC2_P_IN1_DQ",
            # "ASC-PRC2_Y_IN1_DQ",
            # "ASC-SRC1_P_IN1_DQ",
            # "ASC-SRC1_Y_IN1_DQ",
            # "ASC-SRC2_P_IN1_DQ",
            # "ASC-SRC2_Y_IN1_DQ",
            # "ASC-DHARD_P_IN1_DQ",
            # "ASC-DHARD_Y_IN1_DQ",
            # "ASC-CHARD_P_IN1_DQ",
            # "ASC-CHARD_Y_IN1_DQ",
            # "ASC-DSOFT_P_IN1_DQ",
            # "ASC-DSOFT_Y_IN1_DQ",
            # "ASC-CSOFT_P_IN1_DQ",
            # "ASC-CSOFT_Y_IN1_DQ",
        ],
    )
    L1 = Coupling(
        58,
        62,
        [
            "PEM-CS_MAINSMON_EBAY_1_DQ",
            # "ASC-CHARD_P_IN1_DQ",
            # "ASC-CHARD_Y_IN1_DQ",
            # "ASC-CSOFT_P_IN1_DQ",
            # "ASC-DHARD_P_IN1_DQ",
            # "ASC-DHARD_Y_IN1_DQ",
            # "ASC-DSOFT_P_IN1_DQ",
            # "ASC-INP1_P_IN1_DQ",
            # "ASC-MICH_P_IN1_DQ",
            # "ASC-MICH_Y_IN1_DQ",
            # "ASC-PRC1_P_IN1_DQ",
            # "ASC-PRC1_Y_IN1_DQ",
            # "ASC-PRC2_P_IN1_DQ",
            # "ASC-PRC2_Y_IN1_DQ",
            # "ASC-SRC1_P_IN1_DQ",
            # "ASC-SRC1_Y_IN1_DQ",
            # "ASC-SRC2_P_IN1_DQ",
            # "ASC-SRC2_Y_IN1_DQ",
        ],
    )
