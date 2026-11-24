"""JSesh .gly output: the default header and text blocks. See ``trilinear`` for the layout:

    |R11-...-D54-|R12-D21:Q3-...-!
    +tist r=f sb.n Hm=f mSa r tA timHw+s-!
    +lnow his majesty had sent an army to the land of the libyans+s-!
"""

from __future__ import annotations

# JSesh's default document settings, as in the Sinuhe .gly files.
HEADER = """++JSesh_Info 1.0 +s
++JSesh_small_skip 2.0 +s
++JSesh_column_skip 8.0 +s
++JSesh_small_body_scale_limit 12.0 +s
++JSesh_cartouche_line_width 1.0 +s
++JSesh_small_sign_centered false +s
++JSesh_page_direction LEFT_TO_RIGHT +s
++JSesh_max_quadrantHeight 18.0 +s
++JSesh_use_lines_for_shading false +s
++JSesh_max_quadrant_width 22.0 +s
++JSesh_line_skip 6.0 +s
++JSesh_alphabetic_text_baseline 4.0 +s
++JSesh_standard_sign_height 18.0 +s
++JSesh_page_orientation HORIZONTAL +s
"""


def text_block(tag: str, text: str) -> str:
    # "+s" would end the block early; JSesh has no escape for it.
    return f"+{tag}{text.replace('+s', '+ s')}+s-!"
