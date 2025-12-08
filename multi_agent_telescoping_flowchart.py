#!/usr/bin/env python3
"""
Timeline flowchart showing the multi-agent telescoping problem.
Shows dig/dump cycles on horizontal time axes with potential values.
"""

def create_timeline_flowchart():
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="1400" height="1000" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title { font-family: Arial, sans-serif; font-size: 26px; font-weight: bold; fill: #000; }
      .agent-label { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; }
      .time-label { font-family: Arial, sans-serif; font-size: 12px; fill: #666; }
      .event-label { font-family: Arial, sans-serif; font-size: 11px; font-weight: bold; }
      .potential-label { font-family: 'Courier New', monospace; font-size: 10px; fill: #000; }
      .baseline-label { font-family: 'Courier New', monospace; font-size: 10px; fill: #8B0000; font-weight: bold; }
      .world-label { font-family: Arial, sans-serif; font-size: 12px; font-weight: bold; }
      .explanation { font-family: Arial, sans-serif; font-size: 11px; fill: #333; }
      .formula { font-family: 'Courier New', monospace; font-size: 11px; fill: #000; }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#000"/>
    </marker>
    <marker id="arrowhead-red" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#E24A4A"/>
    </marker>
    <marker id="arrowhead-orange" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#FF6B00"/>
    </marker>
  </defs>
  
  <!-- Background -->
  <rect width="1400" height="1000" fill="#fafafa"/>
  
  <!-- Title -->
  <text x="700" y="40" class="title" text-anchor="middle">Multi-Agent Telescoping Problem: Timeline View</text>
  
  <!-- Time axis labels (top) -->
  <text x="200" y="100" class="time-label" text-anchor="middle">t₁</text>
  <text x="400" y="100" class="time-label" text-anchor="middle">t₂</text>
  <text x="600" y="100" class="time-label" text-anchor="middle">t₃</text>
  <text x="800" y="100" class="time-label" text-anchor="middle">t₄</text>
  <text x="1000" y="100" class="time-label" text-anchor="middle">t₅</text>
  <text x="1200" y="100" class="time-label" text-anchor="middle">t₆</text>
  
  <!-- Vertical time markers -->
  <line x1="200" y1="110" x2="200" y2="850" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="400" y1="110" x2="400" y2="850" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="600" y1="110" x2="600" y2="850" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="800" y1="110" x2="800" y2="850" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="1000" y1="110" x2="1000" y2="850" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="1200" y1="110" x2="1200" y2="850" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>
  
  <!-- Agent 1 Timeline -->
  <text x="80" y="180" class="agent-label" text-anchor="middle" fill="#4A90E2">Agent 1</text>
  <line x1="150" y1="180" x2="1250" y2="180" stroke="#4A90E2" stroke-width="4"/>
  
  <!-- Agent 1: DIG at t₁ -->
  <rect x="150" y="150" width="100" height="60" rx="5" fill="#4A90E2" stroke="#000" stroke-width="2"/>
  <text x="200" y="175" class="event-label" text-anchor="middle" fill="white">DIG</text>
  <text x="200" y="190" class="potential-label" text-anchor="middle" fill="white">baseline₁ = 100</text>
  <text x="200" y="205" class="potential-label" text-anchor="middle" fill="white">world_pot = 100</text>
  
  <!-- Agent 1: CARRY from t₁ to t₄ -->
  <line x1="250" y1="180" x2="750" y2="180" stroke="#4A90E2" stroke-width="6" opacity="0.5"/>
  <text x="500" y="175" class="explanation" text-anchor="middle" fill="#4A90E2" font-style="italic">CARRY</text>
  
  <!-- Agent 1: Baseline Update Box 1 (over Agent 2's DIG at t₂) -->
  <rect x="350" y="150" width="100" height="60" rx="5" fill="#FFD700" stroke="#000" stroke-width="2"/>
  <text x="400" y="170" class="event-label" text-anchor="middle" fill="black" font-size="10">Baseline</text>
  <text x="400" y="185" class="event-label" text-anchor="middle" fill="black" font-size="10">Update 1</text>
  <text x="400" y="200" class="potential-label" text-anchor="middle" fill="black" font-size="9">baseline_eff₁</text>
  
  <!-- Agent 1: Baseline Update Box 2 (over Agent 2's DUMP at t₃) -->
  <rect x="550" y="150" width="100" height="60" rx="5" fill="#FFD700" stroke="#000" stroke-width="2"/>
  <text x="600" y="170" class="event-label" text-anchor="middle" fill="black" font-size="10">Baseline</text>
  <text x="600" y="185" class="event-label" text-anchor="middle" fill="black" font-size="10">Update 2</text>
  <text x="600" y="200" class="potential-label" text-anchor="middle" fill="black" font-size="9">baseline_eff₂ = 70</text>
  
  <!-- Arrow from baseline update 1 to baseline update 2 -->
  <line x1="450" y1="180" x2="550" y2="180" stroke="#000" stroke-width="2" stroke-dasharray="3,3" marker-end="url(#arrowhead)"/>
  
  <!-- Arrow from baseline update 2 to Agent 1's dump -->
  <line x1="650" y1="180" x2="700" y2="180" stroke="#000" stroke-width="2" stroke-dasharray="3,3" marker-end="url(#arrowhead)"/>
  
  <!-- Agent 1: DUMP at t₄ -->
  <rect x="700" y="150" width="100" height="60" rx="5" fill="#4A90E2" stroke="#000" stroke-width="2"/>
  <text x="750" y="175" class="event-label" text-anchor="middle" fill="white">DUMP</text>
  <text x="750" y="190" class="potential-label" text-anchor="middle" fill="white">world_pot = 70</text>
  <text x="750" y="205" class="baseline-label" text-anchor="middle" fill="white">reward₁ = ?</text>
  
  <!-- Agent 2 Timeline -->
  <text x="80" y="300" class="agent-label" text-anchor="middle" fill="#E24A4A">Agent 2</text>
  <line x1="150" y1="300" x2="1250" y2="300" stroke="#E24A4A" stroke-width="4"/>
  
  <!-- Agent 2: DIG at t₂ -->
  <rect x="350" y="270" width="100" height="60" rx="5" fill="#E24A4A" stroke="#000" stroke-width="2"/>
  <text x="400" y="295" class="event-label" text-anchor="middle" fill="white">DIG</text>
  <text x="400" y="310" class="potential-label" text-anchor="middle" fill="white">baseline₂ = 95</text>
  <text x="400" y="325" class="potential-label" text-anchor="middle" fill="white">world_pot = 95</text>
  
  <!-- Agent 2: CARRY from t₂ to t₃ -->
  <line x1="450" y1="300" x2="550" y2="300" stroke="#E24A4A" stroke-width="6" opacity="0.5"/>
  <text x="500" y="295" class="explanation" text-anchor="middle" fill="#E24A4A" font-style="italic">CARRY</text>
  
  <!-- Agent 2: DUMP at t₃ -->
  <rect x="550" y="270" width="100" height="60" rx="5" fill="#E24A4A" stroke="#000" stroke-width="2"/>
  <text x="600" y="295" class="event-label" text-anchor="middle" fill="white">DUMP</text>
  <text x="600" y="310" class="potential-label" text-anchor="middle" fill="white">world_pot = 75</text>
  <text x="600" y="325" class="baseline-label" text-anchor="middle" fill="white">reward₂ = 20</text>
  
  <!-- World State Timeline -->
  <text x="80" y="420" class="agent-label" text-anchor="middle" fill="#7ED321">World State</text>
  <text x="80" y="440" class="explanation" text-anchor="middle" fill="#666" font-size="10">Potential</text>
  <line x1="150" y1="420" x2="1250" y2="420" stroke="#7ED321" stroke-width="3"/>
  
  <!-- World potential values -->
  <circle cx="200" cy="420" r="5" fill="#7ED321"/>
  <text x="200" y="410" class="potential-label" text-anchor="middle">100</text>
  
  <circle cx="400" cy="420" r="5" fill="#7ED321"/>
  <text x="400" y="410" class="potential-label" text-anchor="middle">95</text>
  
  <circle cx="600" cy="420" r="5" fill="#7ED321"/>
  <text x="600" y="410" class="potential-label" text-anchor="middle">75</text>
  
  <circle cx="800" cy="420" r="5" fill="#7ED321"/>
  <text x="800" y="410" class="potential-label" text-anchor="middle">70</text>
  
  <!-- Arrows showing world changes -->
  <line x1="200" y1="420" x2="400" y2="420" stroke="#7ED321" stroke-width="2" marker-end="url(#arrowhead)"/>
  <text x="300" y="415" class="explanation" text-anchor="middle" fill="#7ED321" font-size="9">Agent 2 digs</text>
  
  <line x1="400" y1="420" x2="600" y2="420" stroke="#7ED321" stroke-width="2" marker-end="url(#arrowhead)"/>
  <text x="500" y="415" class="explanation" text-anchor="middle" fill="#7ED321" font-size="9">Agent 2 dumps</text>
  
  <line x1="600" y1="420" x2="800" y2="420" stroke="#7ED321" stroke-width="2" marker-end="url(#arrowhead)"/>
  <text x="700" y="415" class="explanation" text-anchor="middle" fill="#7ED321" font-size="9">Agent 1 dumps</text>
  
  <!-- Problem Explanation Box -->
  <rect x="50" y="500" width="1300" height="180" rx="10" fill="#FFF3CD" stroke="#F5A623" stroke-width="3"/>
  <text x="700" y="530" class="world-label" text-anchor="middle" fill="#8B0000" font-size="14">THE PROBLEM: Agent 1's Reward Calculation</text>
  
  <text x="100" y="560" class="explanation" text-anchor="start" fill="#000">Without adjustment:</text>
  <text x="100" y="580" class="formula" text-anchor="start" fill="#000">  reward₁ = baseline₁ - world_pot(t₄)</text>
  <text x="100" y="600" class="formula" text-anchor="start" fill="#000">  reward₁ = 100 - 70 = 30</text>
  <text x="100" y="620" class="explanation" text-anchor="start" fill="#8B0000" font-weight="bold">  ❌ Problem: Agent 1 gets credit for Agent 2's work!</text>
  <text x="100" y="640" class="explanation" text-anchor="start" fill="#000">  Agent 2 changed world from 95 → 75, but Agent 1's reward includes this change</text>
  
  <text x="700" y="560" class="explanation" text-anchor="start" fill="#000">What actually happened:</text>
  <text x="700" y="580" class="explanation" text-anchor="start" fill="#000">  • At t₁: Agent 1 digs, world_pot = 100, baseline₁ = 100</text>
  <text x="700" y="600" class="explanation" text-anchor="start" fill="#000">  • At t₂: Agent 2 digs, world_pot = 95 (Agent 2's action)</text>
  <text x="700" y="620" class="explanation" text-anchor="start" fill="#000">  • At t₃: Agent 2 dumps, world_pot = 75 (Agent 2's action)</text>
  <text x="700" y="640" class="explanation" text-anchor="start" fill="#000">  • At t₄: Agent 1 dumps, world_pot = 70 (Agent 1's action)</text>
  <text x="700" y="660" class="explanation" text-anchor="start" fill="#000" font-weight="bold">  Agent 1 should only get credit for: 75 → 70 = 5 improvement</text>
  
  <!-- Solution Box -->
  <rect x="50" y="710" width="1300" height="200" rx="10" fill="#E8F5E9" stroke="#4CAF50" stroke-width="3"/>
  <text x="700" y="740" class="world-label" text-anchor="middle" fill="#2E7D32" font-size="14">SOLUTION: Effective Baseline Adjustment</text>
  
  <text x="100" y="770" class="explanation" text-anchor="start" fill="#000" font-weight="bold">Track world changes during carry:</text>
  <text x="100" y="790" class="formula" text-anchor="start" fill="#000">  current_potential = world_pot(t₄) = 70</text>
  <text x="100" y="810" class="formula" text-anchor="start" fill="#000">  after_lift_potential = world_pot(t₁ after dig) = 100</text>
  <text x="100" y="830" class="formula" text-anchor="start" fill="#000">  world_change = current_potential - after_lift = 70 - 100 = -30</text>
  
  <text x="700" y="770" class="explanation" text-anchor="start" fill="#000" font-weight="bold">Adjust baseline to account for other agents:</text>
  <text x="700" y="790" class="formula" text-anchor="start" fill="#2E7D32">  baseline_eff = baseline_before + (current_potential - after_lift)</text>
  <text x="700" y="810" class="formula" text-anchor="start" fill="#2E7D32">  baseline_eff = 100 + (70 - 100) = 100 - 30 = 70</text>
  <text x="700" y="830" class="formula" text-anchor="start" fill="#2E7D32">  reward₁ = baseline_eff - new_potential = 70 - 70 = 0</text>
  <text x="700" y="850" class="explanation" text-anchor="start" fill="#2E7D32" font-weight="bold">  ✅ Correct! Agent 1 gets no credit (world was already at 70 from Agent 2's work)</text>
  
  <!-- Alternative scenario showing positive contribution -->
  <rect x="50" y="930" width="1300" height="60" rx="10" fill="#E3F2FD" stroke="#2196F3" stroke-width="2"/>
  <text x="700" y="955" class="explanation" text-anchor="middle" fill="#000" font-weight="bold">Alternative: If Agent 1's dump improves world from 75 → 65:</text>
  <text x="400" y="975" class="formula" text-anchor="start" fill="#000">  baseline_eff = 100 + (65 - 100) = 65</text>
  <text x="400" y="990" class="formula" text-anchor="start" fill="#000">  reward₁ = 65 - 65 = 0 (still correct - no net change from baseline)</text>
  <text x="900" y="975" class="formula" text-anchor="start" fill="#000">  But if world was 75 before Agent 1's dump:</text>
  <text x="900" y="990" class="formula" text-anchor="start" fill="#2E7D32">  reward₁ = 75 - 65 = 10 ✅ (Agent 1 gets credit for 75→65 improvement)</text>
  
  <!-- Dotted arrows showing Agent 2's influence on Agent 1's baseline -->
  <!-- Agent 2 DIG affects Agent 1's baseline calculation -->
  <line x1="400" y1="300" x2="400" y2="210" stroke="#E24A4A" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrowhead-red)"/>
  <text x="410" y="255" class="explanation" text-anchor="start" fill="#E24A4A" font-size="9" font-weight="bold">Agent 2's DIG</text>
  
  <!-- Agent 2 DUMP affects Agent 1's baseline calculation -->
  <line x1="600" y1="300" x2="600" y2="210" stroke="#E24A4A" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrowhead-red)"/>
  <text x="610" y="255" class="explanation" text-anchor="start" fill="#E24A4A" font-size="9" font-weight="bold">Agent 2's DUMP</text>
  
  <!-- Arrow from world state to Agent 1's baseline adjustment 1 -->
  <line x1="400" y1="420" x2="400" y2="210" stroke="#FF6B00" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrowhead-orange)"/>
  <text x="410" y="315" class="explanation" text-anchor="start" fill="#FF6B00" font-size="9" font-weight="bold">World change</text>
  
  <!-- Arrow from world state to Agent 1's baseline adjustment 2 -->
  <line x1="600" y1="420" x2="600" y2="210" stroke="#FF6B00" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#arrowhead-orange)"/>
  <text x="610" y="315" class="explanation" text-anchor="start" fill="#FF6B00" font-size="9" font-weight="bold">World change</text>
  
</svg>'''
    return svg_content

if __name__ == "__main__":
    svg = create_timeline_flowchart()
    output_path = '/cluster/project/rsl/alesweber/TerraProject/terra-baselines/multi_agent_telescoping_flowchart.svg'
    with open(output_path, 'w') as f:
        f.write(svg)
    print(f"Timeline flowchart saved to {output_path}")
    print("The flowchart shows:")
    print("  - Horizontal timelines for Agent 1, Agent 2, and World State")
    print("  - Overlapping dig/dump cycles with time markers")
    print("  - Potential values at each time step")
    print("  - How baseline adjustment solves the credit assignment problem")
