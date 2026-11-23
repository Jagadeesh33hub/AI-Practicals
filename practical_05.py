"""
Practical 5: Rule-Based Expert System with Forward Chaining Inference Engine
Aim: Build a forward-chaining production rule engine that derives intermediate
     hypotheses and final diagnoses with full explanation traces.
"""

from dataclasses import dataclass


@dataclass
class Rule:
    rule_id: str
    conditions: set[str]
    conclusion: str
    recommendation: str


class ForwardChainingExpertSystem:
    def __init__(self, rules: list[Rule]) -> None:
        self.rules = rules

    def infer(self, initial_facts: set[str]) -> tuple[set[str], list[str], list[str]]:
        known_facts = set(initial_facts)
        fired_rules = set()
        trace = []
        recommendations = []

        while True:
            new_fact_added = False
            for rule in self.rules:
                if rule.rule_id not in fired_rules and rule.conditions.issubset(known_facts):
                    fired_rules.add(rule.rule_id)
                    if rule.conclusion not in known_facts:
                        known_facts.add(rule.conclusion)
                        new_fact_added = True
                        trace.append(
                            f"[{rule.rule_id}] IF ({' AND '.join(sorted(rule.conditions))}) "
                            f"-> THEN '{rule.conclusion}'"
                        )
                        if rule.recommendation:
                            recommendations.append(f"{rule.conclusion}: {rule.recommendation}")
            if not new_fact_added:
                break

        return known_facts, trace, recommendations


if __name__ == "__main__":
    KNOWLEDGE_BASE = [
        Rule("R1", {"high_fever", "cough", "body_ache"}, "viral_flu_suspected",
             "Rest, hydrate, monitor temperature, and consult a physician."),
        Rule("R2", {"sneezing", "runny_nose", "itchy_eyes"}, "allergic_rhinitis",
             "Avoid allergen triggers and consider an OTC antihistamine."),
        Rule("R3", {"viral_flu_suspected", "shortness_of_breath"}, "urgent_respiratory_eval",
             "PRIORITY: Seek immediate clinical evaluation and pulse oximetry."),
        Rule("R4", {"sore_throat", "fever", "swollen_tonsils"}, "strep_throat_suspected",
             "Schedule a rapid strep throat swab test."),
        Rule("R5", {"headache", "screen_fatigue", "dry_eyes"}, "digital_eye_strain",
             "Follow the 20-20-20 rule and adjust monitor brightness."),
    ]

    engine = ForwardChainingExpertSystem(KNOWLEDGE_BASE)

    patients = [
        ("Patient A", {"high_fever", "cough", "body_ache", "shortness_of_breath"}),
        ("Patient B", {"sneezing", "runny_nose", "itchy_eyes"}),
        ("Patient C", {"headache", "screen_fatigue", "dry_eyes"}),
    ]

    print("=" * 74)
    print("RULE-BASED MEDICAL ADVISORY EXPERT SYSTEM (FORWARD CHAINING)")
    print("=" * 74)

    for name, symptoms in patients:
        print(f"\n>>> Case: {name}")
        print(f"    Observed Symptoms : {sorted(symptoms)}")
        facts, trace, advice = engine.infer(symptoms)
        print("    Inference Trace   :")
        for step in trace:
            print(f"      {step}")
        print("    Advisory Output   :")
        for item in advice:
            print(f"      * {item}")
