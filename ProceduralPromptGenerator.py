import random

class ProceduralPromptGenerator:
    def __init__(self):
        self.base_positive = "photorealistic, highly detailed, 8k resolution, seamless integration, natural lighting, matching the surrounding urban context"

        # 2. Doors
        self.door_conditions = ["pristine", "weathered", "peeling painted", "rusty", "heavy-duty", "elegant"]
        self.door_materials = ["solid oak", "corrugated metal", "frosted glass", "black aluminum", "cherry-red wooden", "dark iron"]
        self.door_styles = ["modern commercial", "classic arched", "industrial roll-up", "residential front", "vintage antique"]
        
        # 3. Windows
        self.window_styles = ["modern seamless glass", "boarded-up", "heavily barred", "classic bay", "tall industrial"]
        self.window_frames = ["dark metallic frames", "rough plywood sheets", "heavy black iron security grates", "white vinyl trim", "rusted steel frames"]
        self.window_details = ["reflecting the surrounding street", "with visible rusty nails", "with closed white plantation shutters", "with drawn curtains", "shattered and broken"]

        # 4. Facades & Walls
        self.wall_conditions = ["clean and modern", "heavily weathered", "raw and unfinished", "newly painted", "crumbling"]
        self.wall_materials = ["red brick masonry", "matte terracotta stucco", "brutalist concrete", "horizontal cedar wood siding", "white plaster"]
        self.wall_details = ["with highly detailed mortar lines", "with smooth architectural texture", "with visible formwork marks and water stains", "with natural wood grain", "with exposed structural elements"]

        # 5. Graffiti & Decals
        self.graffiti_styles = ["A massive, vibrant geometric", "Messy, overlapping", "A heavily faded, peeling", "A chaotic layer of torn", "A highly detailed stencil"]
        self.graffiti_types = ["street art mural", "spray paint tags in black and silver", "vintage painted advertisement from the 1950s", "paper street posters and concert flyers", "underground graffiti masterpiece"]
        self.graffiti_details = ["painted directly on the wall", "with dripping paint texture", "barely visible with distressed texture", "with peeling edges and urban grunge", "covering the brickwork"]

    def _build_prompt(self, components):
        """Helper to combine random choices into a single string."""
        selected = [random.choice(comp) for comp in components]
        # Combine the selected components and append the universal photorealism anchors
        prompt = " ".join(selected) + ", " + self.base_positive
        return prompt

    def get_prompt(self, category):
        """Returns a tuple of (positive_prompt, negative_prompt) based on the category."""
        category = category.lower()
        
        if category == "building doors":
            pos = self._build_prompt([["A"], self.door_conditions, self.door_materials, self.door_styles, ["door"]])
            
        elif category == "building windows":
            pos = self._build_prompt([["A set of"], self.window_styles, ["windows with"], self.window_frames, ["and"], self.window_details])
            
        elif category in ["walls", "facade"]:
            if random.choice([True, False]):
                pos = self._build_prompt([self.graffiti_styles, self.graffiti_types, self.graffiti_details])
            else:
                pos = self._build_prompt([["A"], self.wall_conditions, self.wall_materials, ["exterior wall"], self.wall_details])
            
        else:
            raise ValueError(f"Unknown category: {category}. Choose from: tree, door, window, wall, graffiti.")
            
        return pos