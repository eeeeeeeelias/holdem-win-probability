THREE_COMMUNITY_CARDS_TESTS = [
    [
        [('AH', '2D'), ('8S', '6D'), ('6H', 'KD'), ('QH', '2C'), ('QC', '4D')],
        ['QS', '3D', '7C'],
        ['3C', '10D', '2S', '10S', '9H', '6S', '5C', 'AD', '9C', 'AS', '4H', 'KS', 'JC', 'AC', '10H', '7D', '7S'],
        [0.0, 0.08658008658008658, 0.14285714285714285, 0.3354978354978355, 0.43506493506493504]
    ],
                
    [
        [('2H', '10H'), ('9C', '9D'), ('4D', '8C'), ('10C', 'QS')],
        ['7D', '7C', 'AD'],
        ['AC', '8D', 'QC', 'QH', '2S', '4H', 'AH', '10D', '5D', 'JS', '5H', '10S', '2C', 'KC', '9S', '3H', '7H'],
        [0.0009057971014492754, 0.8704710144927537, 0.03713768115942029, 0.09148550724637682]
    ],
                
    [
        [('2H', '7D'), ('KS', '8S'), ('JS', '9S')],
        ['2D', '7S', '4C'],
        ['9D', '6S', 'KC', 'AH', '3D', '5H', '3S', '3C', '8C', '7C', '9C', '5D', '6D', '8D', '7H', 'JD', '4D', '8H'],
        [0.9066666666666666, 0.06333333333333334, 0.03]
    ],
                
    [
        [('7D', 'KC'), ('6D', '4C'), ('JD', '2H'), ('4H', '2C')],
        ['5D', '3D', 'AH'],
        ['5H', 'KD', '6C', '8S', 'AS', '6S', 'KH', '10C', '4S', '8H', 'QC', 'QS', '8D', '3S', '2D', '5C', '9D', '6H'],
        [0.0, 0.3241106719367589, 0.0533596837944664, 0.6225296442687747]
    ],
                
    [
        [('QS', '2C'), ('QC', 'JS')],
        ['JH', '2S', '10H'],
        ['AC', '7D', 'QD', '4C', 'KH', '5D', '2D', 'AH', '4S', '10C', '6C', '6S', '9H', '10D', 'QH', '9S', '8C', 'KS', '4D', '8D'],
        [0.09333333333333334, 0.9066666666666666]
    ],
                
    [
        [('9C', '2S'), ('3H', '4D'), ('7D', 'JS'), ('4C', '5H')],
        ['QH', '6C', 'JH'],
        ['9H', '8S', 'KS', '6S', '10S', '10D', '5S', '2D', 'KH', '5C', 'QD', '7C', '3D', 'AD', '7H', '6H', '4H', '10C', '2H', '6D', 'QC', 'JD'],
        [0.04093567251461988, 0.029239766081871343, 0.8654970760233918, 0.06432748538011696]
    ],
                
    [
        [('KC', '9S'), ('10H', 'AD')],
        ['8S', '4D', '5D'],
        ['6D', '6C', '2D', '5H', '8D', 'QC', '7H', '3H', '3S', 'JC', '7S', 'AS', 'QD', 'KS', '2C', '9D', '6S', 'QS', '2S', '5S', '3D'],
        [0.2572463768115942, 0.7427536231884058]
    ],
                
    [
        [('4C', '5H'), ('7D', 'QS'), ('KH', '8S')],
        ['QH', '3H', 'AD'],
        ['4D', '3D', 'AH', '10H', '7S', '8D', '3C', '2C', '8C', 'KC', '7H', '4H', '5S', 'QD', '2D', '8H', '7C', 'QC'],
        [0.15666666666666668, 0.64, 0.20333333333333334]
    ],
                
    [
        [('2S', 'QD'), ('QC', '10H')],
        ['9S', 'KS', '6D'],
        ['8C', '7C', 'JD', 'KC', '10C', 'JC', '3D', '8S', 'AH', '3C', '5D', '4C', '5H', '2H', '4H', '8H', '10S', '6S', '5S', 'KD'],
        [0.25666666666666665, 0.7433333333333333]
    ],
                
    [
        [('10D', 'JS'), ('4H', '10H'), ('KS', 'AS'), ('KD', '8D'), ('5H', 'KC')],
        ['9H', '10S', '10C'],
        ['QC', '3S', '9D', '7S', 'AH', '5D', '6S', '6H', '8C', 'KH', 'JC', 'AD', '6D', '5S', 'QS', '2S', '8H', 'JD', '7C', '4S', '6C', '2C', '3C'],
        [0.625, 0.375, 0.0, 0.0, 0.0]
    ],
                
    [
        [('3H', '8S'), ('8C', 'JS'), ('JH', 'AD'), ('7D', 'AH')],
        ['2H', '5D', '6S'],
        ['9D', '6C', '6D', '3D', '6H', '8H', 'AC', 'QS', '3S', '5S', '2S', '8D', '9H', '10H', '4S', '2C'],
        [0.265, 0.025, 0.5366666666666666, 0.17333333333333334]
    ],
                
    [
        [('3H', '4S'), ('JC', 'QS')],
        ['10H', '5D', '9H'],
        ['10S', '9D', 'JD', '8D', '2H', '3D', 'KS', 'KC', '6S', '8S', 'AC', '9C', '7S', '6H', 'KH', '3S', 'QC', '8C', '7C', 'AD'],
        [0.3333333333333333, 0.6666666666666666]
    ],
                
    [
        [('3C', 'QH'), ('6S', 'QS'), ('6C', '10D'), ('3D', '8C'), ('10C', '5H')],
        ['6D', '2C', '2S'],
        ['KS', '4C', 'JC', 'AC', '8S', 'AS', 'AD', 'KH', 'QC', '9H', '4S', '6H', '5D', '7C', '7D', '3S', '5C', 'JD', '3H', '5S', '2D', '7S', '9S', 'AH'],
        [0.004761904761904762, 0.45714285714285713, 0.32857142857142857, 0.2, 0.009523809523809525]
    ],
                
    [
        [('4C', 'KC'), ('4H', '7D'), ('AD', 'KH'), ('AC', '6H')],
        ['8C', '5D', '5C'],
        ['8D', 'JS', 'QS', '9C', '5S', '6C', '10S', 'KS', '6D', 'JC', '3S', 'QC', '9H', 'JD', '3D', '7H', '2D', '10H', '8S', '10C', '2H', 'AS', '9S', '8H'],
        [0.41544117647058826, 0.16176470588235295, 0.3639705882352941, 0.058823529411764705]
    ],
                
    [
        [('5C', '8S'), ('6C', '6H')],
        ['AD', '3H', '3S'],
        ['3C', '6S', '9S', 'AS', 'KS', 'QD', '8C', 'JH', '2H', '9D', '3D', 'JS', '10C', '7D', '5D', '4C', 'AH', 'JC', 'QC'],
        [0.1753846153846154, 0.8246153846153846]
    ],
                
    [
        [('KD', '5C'), ('JH', '7H'), ('7D', 'JD'), ('9S', '4S')],
        ['6D', '5D', '6C'],
        ['KC', '10D', 'KS', 'QH', 'AD', '2S', '7C', '6H', '3S', '8S', '7S', '2H', '8H', '4C', '5S', '10H', 'JC', 'QC', '4D', 'AH'],
        [0.4583333333333333, 0.04404761904761905, 0.3726190476190476, 0.125]
    ],
                
    [
        [('9H', 'QC'), ('6C', '5C')],
        ['QH', 'JC', '4H'],
        ['2D', '5D', '3S', '10H', '6D', 'QD', '8H', '9C', '4S', '8D', 'AD', '10D', 'QS', '5H', '8C', '2C', 'KS', '4C', '9D', 'KD', '4D', '3C', '7S', '7D'],
        [0.919047619047619, 0.08095238095238096]
    ],
                
    [
        [('AS', '4C'), ('4S', 'AC')],
        ['3D', '2H', '10H'],
        ['3S', 'JC', '5C', '4D', '10C', '2C', 'JH', 'JD', '9C', '5H', 'QH', '9H', '6D', '10S', 'QS', '9S', '7S', '4H'],
        [0.5, 0.5]
    ],
                
    [
        [('6H', '8C'), ('QD', '8D'), ('3D', 'JS'), ('AC', 'AH')],
        ['4D', '7C', 'KH'],
        ['8H', '7H', 'KS', 'AD', '3H', 'QC', '6D', 'KC', '7S', '7D', '3C', '9D', '3S', '8S', '9H', 'JH', '10S', 'JD', '9C', '6S', '10D', '4H', '10H', '4C'],
        [0.40441176470588236, 0.04411764705882353, 0.0, 0.5514705882352942]
    ],
                
]
