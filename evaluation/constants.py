import cmudict

NO_OF_PRECEDING_LINES = 3

NOT_AVAILABLE = 'X'
IPA_VOWELS = {'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
              'ɪ', 'ʏ', 'ɪ̈', 'ʊ̈', 'ʊ',
              'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
              'ə',
              'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
              'æ', 'ɐ',
              'a', 'ɶ', 'ɑ', 'ɒ'}
# Used in CMUdict.
ARPA_VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
               'EH', 'ER', 'EY',
               'IH', 'IY',
               'OW', 'OY',
               'UH', 'UW'}
                    # Differing in place of aritculation only (voiceless plosives, fricatives, nasals).
SIMILAR_SOUNDS = [['M', 'N', 'NG'], ['P', 'T', 'K', 'Q'], ['F', 'TH', 'S', 'SH', 'HH'], ['DH', 'V', 'Z', 'ZH'],
                  # Differing in place only (voiced plosives).
                  ['B', 'D', 'G'],
                  # Differing in voice only (plosives, fricatives).
                  ['B', 'P'], ['T', 'D'], ['K', 'G'], ['F', 'V'], ['TH', 'DH'], ['S', 'Z'], ['SH', 'ZH'],
                  # Differing in continuacy only (coronals).
                  ['T', 'TH'], ['P', 'F'], ['K', 'X'],
                  # Differing in height.
                  ['AA', 'AH'], ['IY', 'IH'], ['EH', 'AE'], ['UW', 'UH', 'AO'],
                  # Differing in front/back (+minor difference in height).
                  ['IH', 'UH'], ['EH', 'ER', 'AH', 'AE'],
                  # Diphthongs sharing a vowel.
                  ['AW', 'OW'], ['AY', 'EY', 'OY']]
# Differing in place of articulation only (nasals/voiceless plosives/sibilants).
HOLTMAN_SIMILAR_CONS1 = {}
# Differing in height or ATR.
HOLTMAN_SIMILAR_VOW1 = {}
dict = cmudict.dict()
# Examples of input and output.
poem1 = ['Roses are red', 'you are too', "please don't be mad", 'be a fool.']
poem2 = ["Twinkle, twinkle, little star,", "How I wonder what you are.", "Up above the world so high,", "Like a diamond in the sky.", "When the blazing sun is gone,", "When he nothing shines upon,", "Then you show your little light,", "Twinkle, twinkle, all the night."]
lyrics1 = ["We were both young when I first saw you.", "I close my eyes and the flashback starts:", "I'm standing there", "On a balcony in summer air.", "See the lights, see the party, the ball gowns,", "See you make your way through the crowd,", "And say, Hello.", "Little did I know...",
           "That you were Romeo, you were throwing pebbles", "And my daddy said, Stay away from Juliet.", "And I was crying on the staircase", "Begging you, Please don't go.", "And I said,", "Romeo, take me somewhere we can be alone.", "I'll be waiting. All there's left to do is run.",
           "You'll be the prince and I'll be the princess.", "It's a love story. Baby, just say 'Yes'."]
# scheme: a, b, c, c, d, e, f, f, g, h, i, j, k, l, m, n, n
lyrics2 = ["I'm at a party I don't wanna be at", "And I don't ever wear a suit and tie, yeah", "Wonderin' if I could sneak out the back", "Nobody's even lookin' me in my eyes", "Can you take my hand?", "Finish my drink, say, Shall we dance?", "You know I love ya, did I ever tell ya?",
           "You make it better like that", "Don't think I fit in at this party", "Everyone's got so much to say", "I always feel like I'm nobody", "Who wants to fit in anyway?", "Cause I don't care when I'm with my baby, yeah", "All the bad things disappear"]
# a, a, b, a, c, d, a, c, e, f, e, f, g, g
lyrics2_syllables_correct = [['wi', 'wər', 'boʊθ', 'jəŋ', 'wɪn', 'aɪ', 'fərst', 'sɔ', 'ju'],
                             ['aɪ', 'kloʊz', 'maɪ', 'aɪz', 'ənd', 'ðə', 'ˈflæʃ', 'ˌbæk', 'stɑrts'],
                             ['əm', 'ˈstæn', 'dɪŋ', 'ðɛr'],
                             ['ɔn', 'ə', 'ˈbæl', 'kə', 'ni', 'ɪn', 'ˈsə', 'mər', 'ɛr'],
                             ['si', 'ðə', 'laɪts', 'si', 'ðə', 'ˈpɑr', 'ti', 'ðə', 'bɔl', 'gaʊnz'],
                             ['si', 'ju', 'meɪk', 'jʊr', 'weɪ', 'θru', 'ðə', 'kraʊd'],
                             ['ənd', 'seɪ', 'hɛˈ', 'loʊ'],
                             ['ˈlɪ', 'təl', 'dɪd', 'aɪ', 'noʊ'],
                             ['ðət', 'ju', 'wər', 'ˈroʊ', 'mi', 'ˌoʊ', 'ju', 'wər', 'θro', 'ʊɪŋ', 'ˈpɛ', 'bəlz'],
                             ['ənd', 'maɪ', 'ˈdæ', 'di', 'sɛd', 'steɪ', 'əˈ', 'weɪ', 'frəm', 'ˈʤu', 'li', 'ˌɛt'],
                             ['ənd', 'aɪ', 'wɑz', 'kraɪ', 'ɪŋ', 'ɔn', 'ðə', 'ˈstɛr', 'ˌkeɪs'],
                             ['ˈbɛ', 'gɪŋ', 'ju', 'pliz', 'doʊnt', 'goʊ'],
                             ['ənd', 'aɪ', 'sɛd'],
                             ['ˈroʊ', 'mi', 'ˌoʊ', 'teɪk', 'mi', 'ˈsəm', 'ˌwɛr', 'wi', 'kən', 'bi', 'əˈ', 'loʊn'],
                             ['aɪl', 'bi', 'ˈweɪ', 'tɪŋ', 'ɔl', 'ðɛrz', 'lɛft', 'tɪ', 'du', 'ɪz', 'rən'],
                             ['jul', 'bi', 'ðə', 'prɪns', 'ənd', 'aɪl', 'bi', 'ðə', 'ˈprɪn', 'sɛs'],
                             ['ɪts', 'ə', 'ləv', 'ˈstɔ', 'ri', 'ˈbeɪ', 'bi', 'ʤɪst', 'seɪ', 'jɛs']]

