Hej med jer :)

Jeg har struktureret python programmet således at alle billedeanalyse komponenterne er separate.
Dvs. at nu er /src/ folderen blevet til et python modul frem for at være selvstændigt kørende python scripts.

FOR AT KØRE PROGRAMMET:
Skal i køre run.py. Den fil skal aldrig laves om da den starter ud med at køre main.py som så kører resten af billedeanalyse modulet

Det virker lidt indviklet men på den måde følger vi et programmeringsprincip som hedder "low coupling and high cohesion".
Langsigtet vil det gøre kodebasen mere fleksibel og meget nemmere at vedligeholde ^^

"We do this not because it is easy, but because it is hard" - JFK