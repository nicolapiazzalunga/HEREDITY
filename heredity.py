import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Initialize helper variables
    parents = {"father", "mother"}
    jointprobability = 1

    # For each person iterate the sets
    for person in people:

        # If person has one gene
        if person in one_gene:

            # Set gene_selector for computation of probability to have a trait further down in the code
            gene_selector = 1

            # If person has parents
            if has_parents(person, parents, people):

                # Select appropriate probability of passing gene for each parent
                parent_passing_gene = dict()
                for parent in parents:
                    parent_name = people[person][parent]
                    parent_passing_gene[parent] = probability_of_passing_gene(parent_name, one_gene, two_genes)
                
                # Compute probability of only one gene being passed by parents (A XOR B)
                probability_one_gene = (
                    parent_passing_gene["father"] * (1 - parent_passing_gene["mother"]) +
                    parent_passing_gene["mother"] * (1 - parent_passing_gene["father"])
                )

            # If person has no parent                    
            else:
                probability_one_gene = PROBS["gene"][gene_selector]

            # Update joint probability
            jointprobability *= probability_one_gene

        # If person has two genes
        elif person in two_genes:

            # Set gene_selector for computation of probability to have a trait further down in the code
            gene_selector = 2

            # If person has parents
            if has_parents(person, parents, people):
                    
                # Select appropriate probability of passing gene for each parent
                parent_passing_gene = dict()
                for parent in parents:
                    parent_name = people[person][parent]
                    parent_passing_gene[parent] = probability_of_passing_gene(parent_name, one_gene, two_genes)
                
                # Compute probability of two genes being passed by parents (A AND B)
                probability_two_genes = parent_passing_gene["father"] * parent_passing_gene["mother"]
            
            # If person has no parent
            else:
                probability_two_genes = PROBS["gene"][gene_selector]

            # Update joint probability
            jointprobability *= probability_two_genes

        # If person has no gene
        else:

            # Set gene_selector for computation of probability to have a trait further down in the code
            gene_selector = 0

            # If person has parents
            if has_parents(person, parents, people):
                
                # Select appropriate probability of passing gene for each parent
                parent_passing_gene = dict()
                for parent in parents:
                    parent_name = people[person][parent]
                    parent_passing_gene[parent] = probability_of_passing_gene(parent_name, one_gene, two_genes)

                # Compute probability of no gene being passed by parents (NOT A AND NOT B)
                probability_no_gene = (1 - parent_passing_gene["father"]) * (1 - parent_passing_gene["mother"])

            # If person has no parent
            else:
                probability_no_gene = PROBS["gene"][gene_selector]
            
            # Update joint probability
            jointprobability *= probability_no_gene

        # If person has trait
        if person in have_trait:
            probability_has_trait = PROBS["trait"][gene_selector][True]
            jointprobability *= probability_has_trait

        # If person does not have trait
        else:
            probability_no_trait = PROBS["trait"][gene_selector][False]
            jointprobability *= probability_no_trait

    # Return joint probability
    return jointprobability


def has_parents(person, parents, people):
    """
    True if person has a parent, False otherwise
    """
    return any((people[person][parent] is not None) for parent in parents)


def probability_of_passing_gene(parent_name, one_gene, two_genes):
    """
    Returns the probability of parent_name to pass a gene
    """
    if parent_name in one_gene:
        return 0.5
    elif parent_name in two_genes:
        return 1 - PROBS["mutation"]
    else:
        return PROBS["mutation"]


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        
        # Update the 'gene' probabilities
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p
        
        # Update the 'trait' probabilities
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    variables = {"gene", "trait"}
    for person in probabilities:
        for variable in variables:
            normalizing_factor = 0
            for bucket in probabilities[person][variable]:
                normalizing_factor += probabilities[person][variable][bucket]
            for bucket in probabilities[person][variable]:
                probabilities[person][variable][bucket] /= normalizing_factor


if __name__ == "__main__":
    main()
