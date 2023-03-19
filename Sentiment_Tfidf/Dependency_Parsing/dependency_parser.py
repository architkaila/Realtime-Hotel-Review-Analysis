import stanza

def extract_adjectives_with_nouns(review_text, nlp_stanza):
    """
    Extract the adjectives and the nouns they are describing

    Args:
        review_text (str): review text
    
    Returns:
        adj_noun_pairs (list): list of adjective-noun pairs
    """

    ## extract all the adjectives and the nouns they are describing
    doc = nlp_stanza(review_text)
    adj_noun_pairs = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos == "ADJ":
                for child in sentence.words:
                    if child.head == word.id and child.upos == "NOUN":
                        adj_noun_pairs.append((word.text, child.text))

    return adj_noun_pairs

def extract_adjectives_with_dependencies(review_text, nlp_stanza):
    """
    Extract the adjectives and the dependencies they have (noun, pronoun, adverb, etc.)

    Args:
        review_text (str): review text
    
    Returns:
        adjective_dependencies (dict): dictionary of adjectives and their dependencies
    """
    doc = nlp_stanza(review_text)
    adjective_dependencies = {}  # Create an empty dictionary to store the adjective dependencies
    for sent in doc.sentences:  # Loop through each sentence in the parsed document
        for word in sent.words:  # Loop through each word in the sentence
            if word.upos == 'ADJ':  # If the word is an adjective
                adjective = word.text
                parent_word = sent.words[word.head - 1]  # Get the parent word of the adjective
                if parent_word.deprel == 'root':  # If the parent is the root of the tree, don't include it
                    continue
                dependencies = [parent_word.text]  # Initialize a list of dependencies with the parent word
                for candidate_child in sent.words:
                    if (candidate_child.head-1) == parent_word.id and candidate_child.deprel in ['amod', 'nsubj', 'advmod']:
                        if candidate_child.text != parent_word.text:
                            dependencies.append(candidate_child.text)  # Add the child to the list of dependencies
                        for grandchild in sent.words:  # Loop through the children of the child
                            if (grandchild.head-1) == candidate_child.id and grandchild.deprel in ['amod', 'nsubj', 'advmod']:
                                if grandchild.text != candidate_child.text:
                                    dependencies.append(grandchild.text)  # Add the grandchild to the list of dependencies
                adjective_dependencies[adjective] = list(set(dependencies))  # Add the adjective and its dependencies to the dictionary, removing any duplicates
    return adjective_dependencies


def dependency_parser(review_text, keywords, nlp_stanza):
    """
    Extract the adjectives and the nouns they are describing using the dependency parser

    Args:
        review_text (str): review text
        keywords (list): list of top keywords using shape values
    
    Returns:
        top_adjective_shap (list): list of adjective-noun pairs using shap values
        remaining_adjectives (list): list of adjective-noun pairs using dependency parsing
    """
    
    ## extract all the adjectives and the nouns they are describing
    adj_noun_pairs = extract_adjectives_with_nouns(review_text, nlp_stanza)
    ## extract all the adjectives and the dependencies they have
    adjective_dependencies = extract_adjectives_with_dependencies(review_text, nlp_stanza)

    top_adjective_shap = []
    remaining_adjectives = []

    ## Process the adjectives and nouns
    for adj, noun in adj_noun_pairs:
        if adj in keywords:
            top_adjective_shap.append((adj, noun))
        else:
            remaining_adjectives.append((adj, noun))

    ## Process the adjectives and dependencies
    for adjective, dependencies in adjective_dependencies.items():
        for dependency in dependencies:
            if dependency!=".":
                if adjective in keywords:
                    top_adjective_shap.append((adjective, dependency))
                else:
                    remaining_adjectives.append((adjective, dependency))

    top_adjective_shap = list(set(top_adjective_shap))
    remaining_adjectives = list(set(remaining_adjectives))

    return top_adjective_shap, remaining_adjectives