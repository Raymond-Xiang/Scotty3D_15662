// clang-format off
#include "pipeline.h"

#include <iostream>

#include "../lib/log.h"
#include "../lib/mathlib.h"
#include "framebuffer.h"
#include "sample_pattern.h"
template<PrimitiveType primitive_type, class Program, uint32_t flags>
void Pipeline<primitive_type, Program, flags>::run(std::vector<Vertex> const& vertices,
                                                   typename Program::Parameters const& parameters,
                                                   Framebuffer* framebuffer_) {
	// Framebuffer must be non-null:
	assert(framebuffer_);
	auto& framebuffer = *framebuffer_;

	// A1T7: sample loop
	// TODO: update this function to rasterize to *all* sample locations in the framebuffer.
	//  	 This will probably involve inserting a loop of the form:
	// 		 	std::vector< Vec3 > const &samples = framebuffer.sample_pattern.centers_and_weights;
	//      	for (uint32_t s = 0; s < samples.size(); ++s) { ... }
	//   	 around some subset of the code.
	// 		 You will also need to transform the input and output of the rasterize_* functions to
	// 	     account for the fact they deal with pixels centered at (0.5,0.5).

	std::vector<ShadedVertex> shaded_vertices;
	shaded_vertices.reserve(vertices.size());

	//--------------------------
	// shade vertices:
	for (auto const& v : vertices) {
		ShadedVertex sv;
		Program::shade_vertex(parameters, v.attributes, &sv.clip_position, &sv.attributes);
		shaded_vertices.emplace_back(sv);
	}

	//--------------------------
	// assemble + clip + homogeneous divide vertices:
	std::vector<ClippedVertex> clipped_vertices;

	// reserve some space to avoid reallocations later:
	if constexpr (primitive_type == PrimitiveType::Lines) {
		// clipping lines can never produce more than one vertex per input vertex:
		clipped_vertices.reserve(shaded_vertices.size());
	} else if constexpr (primitive_type == PrimitiveType::Triangles) {
		// clipping triangles can produce up to 8 vertices per input vertex:
		clipped_vertices.reserve(shaded_vertices.size() * 8);
	}
	// clang-format off

	//coefficients to map from clip coordinates to framebuffer (i.e., "viewport") coordinates:
	//x: [-1,1] -> [0,width]
	//y: [-1,1] -> [0,height]
	//z: [-1,1] -> [0,1] (OpenGL-style depth range)
	Vec3 const clip_to_fb_scale = Vec3{
		framebuffer.width / 2.0f,
		framebuffer.height / 2.0f,
		0.5f
	};
	Vec3 const clip_to_fb_offset = Vec3{
		0.5f * framebuffer.width,
		0.5f * framebuffer.height,
		0.5f
	};

	// helper used to put output of clipping functions into clipped_vertices:
	auto emit_vertex = [&](ShadedVertex const& sv) {
		ClippedVertex cv;
		float inv_w = 1.0f / sv.clip_position.w;
		cv.fb_position = clip_to_fb_scale * inv_w * sv.clip_position.xyz() + clip_to_fb_offset;
		cv.inv_w = inv_w;
		cv.attributes = sv.attributes;
		clipped_vertices.emplace_back(cv);
	};

	// actually do clipping:
	if constexpr (primitive_type == PrimitiveType::Lines) {
		for (uint32_t i = 0; i + 1 < shaded_vertices.size(); i += 2) {
			clip_line(shaded_vertices[i], shaded_vertices[i + 1], emit_vertex);
		}
	} else if constexpr (primitive_type == PrimitiveType::Triangles) {
		for (uint32_t i = 0; i + 2 < shaded_vertices.size(); i += 3) {
			clip_triangle(shaded_vertices[i], shaded_vertices[i + 1], shaded_vertices[i + 2], emit_vertex);
		}
	} else {
		static_assert(primitive_type == PrimitiveType::Lines, "Unsupported primitive type.");
	}

	//--------------------------
	// rasterize primitives:

	std::vector<Fragment> fragments;

	// helper used to put output of rasterization functions into fragments:
	auto emit_fragment = [&](Fragment const& f) { fragments.emplace_back(f); };

	// actually do rasterization:
	if constexpr (primitive_type == PrimitiveType::Lines) {
		for (uint32_t i = 0; i + 1 < clipped_vertices.size(); i += 2) {
			rasterize_line(clipped_vertices[i], clipped_vertices[i + 1], emit_fragment);
		}
	} else if constexpr (primitive_type == PrimitiveType::Triangles) {
		for (uint32_t i = 0; i + 2 < clipped_vertices.size(); i += 3) {
			rasterize_triangle(clipped_vertices[i], clipped_vertices[i + 1], clipped_vertices[i + 2], emit_fragment);
		}
	} else {
		static_assert(primitive_type == PrimitiveType::Lines, "Unsupported primitive type.");
	}

	//--------------------------
	// depth test + shade + blend fragments:
	uint32_t out_of_range = 0; // check if rasterization produced fragments outside framebuffer 
							   // (indicates something is wrong with clipping)
	for (auto const& f : fragments) {

		// fragment location (in pixels):
		int32_t x = (int32_t)std::floor(f.fb_position.x);
		int32_t y = (int32_t)std::floor(f.fb_position.y);

		// if clipping is working properly, this condition shouldn't be needed;
		// however, it prevents crashes while you are working on your clipping functions,
		// so we suggest leaving it in place:
		if (x < 0 || (uint32_t)x >= framebuffer.width || 
		    y < 0 || (uint32_t)y >= framebuffer.height) {
			++out_of_range;
			continue;
		}

		// local names that refer to destination sample in framebuffer:
		float& fb_depth = framebuffer.depth_at(x, y, 0);
		Spectrum& fb_color = framebuffer.color_at(x, y, 0);


		// depth test:
		if constexpr ((flags & PipelineMask_Depth) == Pipeline_Depth_Always) {
			// "Always" means the depth test always passes.
		} else if constexpr ((flags & PipelineMask_Depth) == Pipeline_Depth_Never) {
			// "Never" means the depth test never passes.
			continue; //discard this fragment
		} else if constexpr ((flags & PipelineMask_Depth) == Pipeline_Depth_Less) {
			// "Less" means the depth test passes when the new fragment has depth less than the stored depth.
			// A1T4: Depth_Less
			// TODO: implement depth test! We want to only emit fragments that have a depth less than the stored depth, hence "Depth_Less".
			
			// if this fragment is deeper than the depth in frame buffer (i.e. what we have drawn so far)
			// it will not be displayed
			if (f.fb_position.z >= fb_depth) {
        		continue; 
			}
			//otherwise, go ahead
		} else {
			static_assert((flags & PipelineMask_Depth) <= Pipeline_Depth_Always, "Unknown depth test flag.");
		}

		// if depth test passes, and depth writes aren't disabled, write depth to depth buffer:
		if constexpr (!(flags & Pipeline_DepthWriteDisableBit)) {
			fb_depth = f.fb_position.z;
		}

		// shade fragment:
		ShadedFragment sf;
		sf.fb_position = f.fb_position;
		Program::shade_fragment(parameters, f.attributes, f.derivatives, &sf.color, &sf.opacity);

		// write color to framebuffer if color writes aren't disabled:
		if constexpr (!(flags & Pipeline_ColorWriteDisableBit)) {
			// blend fragment:
			if constexpr ((flags & PipelineMask_Blend) == Pipeline_Blend_Replace) {
				fb_color = sf.color;
			} else if constexpr ((flags & PipelineMask_Blend) == Pipeline_Blend_Add) {
				// A1T4: Blend_Add
				// TODO: framebuffer color should have fragment color multiplied by fragment opacity added to it.
				
				//NOTE: DON'T USE += OVER HERE!
				// Blend add: if depth check tells me to display something, I just add it on top of frame buffer
				fb_color = fb_color + sf.color * sf.opacity;
			} else if constexpr ((flags & PipelineMask_Blend) == Pipeline_Blend_Over) {
				// A1T4: Blend_Over
				// TODO: set framebuffer color to the result of "over" blending (also called "alpha blending") the fragment color over the framebuffer color, using the fragment's opacity
				// 		 You may assume that the framebuffer color has its alpha premultiplied already, and you just want to compute the resulting composite color

				//if depth check tells me to display something, I add ...
				//this fragment's color and opacity on top of current colors, but I have blocked some of the previous colors
				fb_color = sf.color * sf.opacity + fb_color * (1.0f - sf.opacity);
			} else {
				static_assert((flags & PipelineMask_Blend) <= Pipeline_Blend_Over, "Unknown blending flag.");
			}
		}
	}
	if (out_of_range > 0) {
		if constexpr (primitive_type == PrimitiveType::Lines) {
			warn("Produced %d fragments outside framebuffer; this indicates something is likely "
			     "wrong with the clip_line function.",
			     out_of_range);
		} else if constexpr (primitive_type == PrimitiveType::Triangles) {
			warn("Produced %d fragments outside framebuffer; this indicates something is likely "
			     "wrong with the clip_triangle function.",
			     out_of_range);
		}
	}
}

// -------------------------------------------------------------------------
// clipping functions

// helper to interpolate between vertices:
template<PrimitiveType p, class P, uint32_t F>
auto Pipeline<p, P, F>::lerp(ShadedVertex const& a, ShadedVertex const& b, float t) -> ShadedVertex {
	ShadedVertex ret;
	ret.clip_position = (b.clip_position - a.clip_position) * t + a.clip_position;
	for (uint32_t i = 0; i < ret.attributes.size(); ++i) {
		ret.attributes[i] = (b.attributes[i] - a.attributes[i]) * t + a.attributes[i];
	}
	return ret;
}

/*
 * clip_line - clip line to portion with -w <= x,y,z <= w, emit vertices of clipped line (if non-empty)
 *  	va, vb: endpoints of line
 *  	emit_vertex: call to produce truncated line
 *
 * If clipping shortens the line, attributes of the shortened line should respect the pipeline's interpolation mode.
 * 
 * If no portion of the line remains after clipping, emit_vertex will not be called.
 *
 * The clipped line should have the same direction as the full line.
 */
template<PrimitiveType p, class P, uint32_t flags>
void Pipeline<p, P, flags>::clip_line(ShadedVertex const& va, ShadedVertex const& vb,
                                      std::function<void(ShadedVertex const&)> const& emit_vertex) {
	// Determine portion of line over which:
	// 		pt = (b-a) * t + a
	//  	-pt.w <= pt.x <= pt.w
	//  	-pt.w <= pt.y <= pt.w
	//  	-pt.w <= pt.z <= pt.w
	// ... as a range [min_t, max_t]:

	float min_t = 0.0f;
	float max_t = 1.0f;

	// want to set range of t for a bunch of equations like:
	//    a.x + t * ba.x <= a.w + t * ba.w
	// so here's a helper:
	auto clip_range = [&min_t, &max_t](float l, float dl, float r, float dr) {
		// restrict range such that:
		// l + t * dl <= r + t * dr
		// re-arranging:
		//  l - r <= t * (dr - dl)
		if (dr == dl) {
			// want: l - r <= 0
			if (l - r > 0.0f) {
				// works for none of range, so make range empty:
				min_t = 1.0f;
				max_t = 0.0f;
			}
		} else if (dr > dl) {
			// since dr - dl is positive:
			// want: (l - r) / (dr - dl) <= t
			min_t = std::max(min_t, (l - r) / (dr - dl));
		} else { // dr < dl
			// since dr - dl is negative:
			// want: (l - r) / (dr - dl) >= t
			max_t = std::min(max_t, (l - r) / (dr - dl));
		}
	};

	// local names for clip positions and their difference:
	Vec4 const& a = va.clip_position;
	Vec4 const& b = vb.clip_position;
	Vec4 const ba = b - a;

	// -a.w - t * ba.w <= a.x + t * ba.x <= a.w + t * ba.w
	clip_range(-a.w, -ba.w, a.x, ba.x);
	clip_range(a.x, ba.x, a.w, ba.w);
	// -a.w - t * ba.w <= a.y + t * ba.y <= a.w + t * ba.w
	clip_range(-a.w, -ba.w, a.y, ba.y);
	clip_range(a.y, ba.y, a.w, ba.w);
	// -a.w - t * ba.w <= a.z + t * ba.z <= a.w + t * ba.w
	clip_range(-a.w, -ba.w, a.z, ba.z);
	clip_range(a.z, ba.z, a.w, ba.w);

	if (min_t < max_t) {
		if (min_t == 0.0f) {
			emit_vertex(va);
		} else {
			ShadedVertex out = lerp(va, vb, min_t);
			// don't interpolate attributes if in flat shading mode:
			if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Flat) {
				out.attributes = va.attributes;
			}
			emit_vertex(out);
		}
		if (max_t == 1.0f) {
			emit_vertex(vb);
		} else {
			ShadedVertex out = lerp(va, vb, max_t);
			// don't interpolate attributes if in flat shading mode:
			if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Flat) {
				out.attributes = va.attributes;
			}
			emit_vertex(out);
		}
	}
}

/*
 * clip_triangle - clip triangle to portion with -w <= x,y,z <= w, emit resulting shape as triangles (if non-empty)
 *  	va, vb, vc: vertices of triangle
 *  	emit_vertex: call to produce clipped triangles (three calls per triangle)
 *
 * If clipping truncates the triangle, attributes of the new vertices should respect the pipeline's interpolation mode.
 * 
 * If no portion of the triangle remains after clipping, emit_vertex will not be called.
 *
 * The clipped triangle(s) should have the same winding order as the full triangle.
 */
template<PrimitiveType p, class P, uint32_t flags>
void Pipeline<p, P, flags>::clip_triangle(
	ShadedVertex const& va, ShadedVertex const& vb, ShadedVertex const& vc,
	std::function<void(ShadedVertex const&)> const& emit_vertex) {
	// A1EC: clip_triangle
	// TODO: correct code!
	emit_vertex(va);
	emit_vertex(vb);
	emit_vertex(vc);
}

// -------------------------------------------------------------------------
// rasterization functions

/*
 * rasterize_line:
 * calls emit_fragment( frag ) for every pixel "covered" by the line (va.fb_position.xy, vb.fb_position.xy).
 *
 *    a pixel (x,y) is "covered" by the line if it exits the inscribed diamond:
 * 
 *        (x+0.5,y+1)
 *        /        \
 *    (x,y+0.5)  (x+1,y+0.5)
 *        \        /
 *         (x+0.5,y)
 *
 *    to avoid ambiguity, we consider diamonds to contain their left and bottom points
 *    but not their top and right points. 
 * 
 * 	  since 45 degree lines breaks this rule, our rule in general is to rasterize the line as if its
 *    endpoints va and vb were at va + (e, e^2) and vb + (e, e^2) where no smaller nonzero e produces 
 *    a different rasterization result. 
 *    We will not explicitly check for 45 degree lines along the diamond edges (this will be extra credit),
 *    but you should be able to handle 45 degree lines in every other case (such as starting from pixel centers)
 *
 * for each such diamond, pass Fragment frag to emit_fragment, with:
 *  - frag.fb_position.xy set to the center (x+0.5,y+0.5)
 *  - frag.fb_position.z interpolated linearly between va.fb_position.z and vb.fb_position.z
 *  - frag.attributes set to va.attributes (line will only be used in Interp_Flat mode)
 *  - frag.derivatives set to all (0,0)
 *
 * when interpolating the depth (z) for the fragments, you may use any depth the line takes within the pixel
 * (i.e., you don't need to interpolate to, say, the closest point to the pixel center)
 *
 * If you wish to work in fixed point, check framebuffer.h for useful information about the framebuffer's dimensions.
 */

static inline int sgnf(float v, float eps = 1e-6f) {
    if (v > eps) return 1;
    if (v < -eps) return -1;
    return 0;
}

template<PrimitiveType p, class P, uint32_t flags>
void Pipeline<p, P, flags>::rasterize_line(
    ClippedVertex const& va, ClippedVertex const& vb,
    std::function<void(Fragment const&)> const& emit_fragment) {
    if constexpr ((flags & PipelineMask_Interp) != Pipeline_Interp_Flat) {
        assert(0 && "rasterize_line should only be invoked in flat interpolation mode.");
    }
    
	//A1T2
    
	//some helper functions first
	// check if point p is inside the inscribed diamond of pixel point (px,py)
	// rule of thumb: to be inside the diamond, its L1-norm can't be more than 0.5 
	auto in_diamond = [](Vec2 p, int px, int py) -> bool {
		const float EPS = 1e-6f;
		float cx = px + 0.5f;
		float cy = py + 0.5f;
		float lx = p.x - cx;
		float ly = p.y - cy;
		float sum = std::abs(lx) + std::abs(ly);

		if (sum < 0.5f - EPS) return true;
		if (sum > 0.5f + EPS) return false;

		if (std::abs(sum - 0.5f) <= EPS) {
			// left boundary (lx == -0.5)
			if (std::abs(lx + 0.5f) <= EPS) return true;
			// bottom boundary (ly == -0.5)
			if (std::abs(ly + 0.5f) <= EPS) return true;
			// otherwise it's top or right boundary -> exclude
			return false;
		}
		return false;
	};

	// AB X AC
	auto orient = [](const Vec2 &a, const Vec2 &b, const Vec2 &c) -> float {
		return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
	};

	// check if point P lies directly on AB
	auto on_segment = [&](const Vec2 &a, const Vec2 &b, const Vec2 &p) -> bool {
		const float EPS = 1e-6f;
		if (std::abs(orient(a,b,p)) > EPS) return false;
		// check bounding box
		if ( (p.x + EPS) < std::min(a.x,b.x) || (p.x - EPS) > std::max(a.x,b.x) ) return false;
		if ( (p.y + EPS) < std::min(a.y,b.y) || (p.y - EPS) > std::max(a.y,b.y) ) return false;
		return true;
	};

	// Check segment-segment intersection (returns true if proper intersection or touching)
	auto seg_intersect = [&](const Vec2 &a, const Vec2 &b, const Vec2 &c, const Vec2 &d) -> bool {
		const float EPS = 1e-6f;
		float o1 = orient(a,b,c);
		float o2 = orient(a,b,d);
		float o3 = orient(c,d,a);
		float o4 = orient(c,d,b);

		int s1 = sgnf(o1, EPS), s2 = sgnf(o2, EPS), s3 = sgnf(o3, EPS), s4 = sgnf(o4, EPS);

		if (s1 * s2 < 0 && s3 * s4 < 0) return true; // proper intersection

		// colinear / endpoint cases:
		if (s1 == 0 && on_segment(a,b,c)) return true;
		if (s2 == 0 && on_segment(a,b,d)) return true;
		if (s3 == 0 && on_segment(c,d,a)) return true;
		if (s4 == 0 && on_segment(c,d,b)) return true;
		return false;
	};

	// Check if our segment-of-interest exits the inscribed diamond
	auto exits_diamond = [&](Vec2 start, Vec2 end, int pixel_x, int pixel_y) -> bool {
		// pixel center
		float cx = pixel_x + 0.5f;
		float cy = pixel_y + 0.5f;

		// quick endpoints-in-diamond test (with tie-rule)
		bool in0 = in_diamond(start, pixel_x, pixel_y);
		bool in1 = in_diamond(end, pixel_x, pixel_y);

		if (in0 != in1) {
			// one in one out -> exit/enter happened
			// I'm not sure if we HAVE TO point outwards in order to "exit" the diamond
			// What if the end point is inside?
			// still call it true here
			return true;
		}
		if (in0 && in1) {
			// both inside -> no exit
			return false;
		}
		// both outside -> check if segment crosses diamond edges
		// diamond vertices (in screen coords)
		Vec2 top   = Vec2(cx, cy + 0.5f);
		Vec2 right = Vec2(cx + 0.5f, cy);
		Vec2 bot   = Vec2(cx, cy - 0.5f);
		Vec2 left  = Vec2(cx - 0.5f, cy);

		// four edges in CCW order: top->right, right->bot, bot->left, left->top
		// If the segment intersects ANY of those edges, then it enters/exits the diamond (covers).
		// For tie-rule: intersections located exactly on top/right edges that are "excluded" still
		// represent a touch; but if the segment only touches at a single excluded point and does not pass through,
		// returning true is conservative but acceptable. If you want strict exclusion of top/right single-touch,
		// add extra checks below.

		if (seg_intersect(start, end, top, right)) return true;
		if (seg_intersect(start, end, right, bot)) return true;
		if (seg_intersect(start, end, bot, left)) return true;
		if (seg_intersect(start, end, left, top)) return true;

		return false;
	};
    
	//Work with start & end points first, they don't work well with Bresenham
    Vec2 start_pos = va.fb_position.xy();
    Vec2 end_pos = vb.fb_position.xy();
    
    int start_pixel_x = static_cast<int>(std::floor(start_pos.x));
    int start_pixel_y = static_cast<int>(std::floor(start_pos.y));

	//Maybe starting point exits that diamond?
    if (exits_diamond(start_pos, end_pos, start_pixel_x, start_pixel_y)) {
        Fragment frag;
        frag.fb_position = Vec3(start_pixel_x + 0.5f, start_pixel_y + 0.5f, va.fb_position.z);
        frag.attributes = va.attributes;
        frag.derivatives.fill(Vec2(0.0f, 0.0f));
        emit_fragment(frag);
    }
    
	//Maybe ending point exits that diamond?
    int end_pixel_x = static_cast<int>(std::floor(end_pos.x));
    int end_pixel_y = static_cast<int>(std::floor(end_pos.y));
    if (exits_diamond(end_pos, start_pos, end_pixel_x, end_pixel_y)) {
        Fragment frag;
        frag.fb_position = Vec3(end_pixel_x + 0.5f, end_pixel_y + 0.5f, vb.fb_position.z);
        frag.attributes = va.attributes;
        frag.derivatives.fill(Vec2(0.0f, 0.0f));
        emit_fragment(frag);
    }
    
	//Every pixel in between - we solve it using Bresenham
    ClippedVertex start = va;
    ClippedVertex end = vb;
    
    if(start.fb_position.x > end.fb_position.x){
        std::swap(start, end);
    }
    
    float dx = end.fb_position.x - start.fb_position.x;
    float dy = end.fb_position.y - start.fb_position.y;
    float dz = end.fb_position.z - start.fb_position.z;
    
    bool steep = std::abs(dy) > std::abs(dx);
    
    if (steep) {
        if (start.fb_position.y > end.fb_position.y) {
            std::swap(start, end);
            dx = -dx;
            dy = -dy;
            dz = -dz;
        }
        
        dx = end.fb_position.x - start.fb_position.x;
        dy = end.fb_position.y - start.fb_position.y;
        dz = end.fb_position.z - start.fb_position.z;
        
        if (dy != 0) {
            int x0 = static_cast<int>(std::floor(start.fb_position.x));
            int y0 = static_cast<int>(std::floor(start.fb_position.y));
            int x1 = static_cast<int>(std::floor(end.fb_position.x));
            int y1 = static_cast<int>(std::floor(end.fb_position.y));
            
            int dx_int = x1 - x0;
            int dy_int = y1 - y0;
            
			//Skip the pixel containing starting and ending points (cuz they're alread dealt with!)
            if (dx_int == 0) {
                for (int y = y0 + 1; y < y1; y++) {
                    Fragment frag;
                    frag.fb_position.x = x0 + 0.5f;
                    frag.fb_position.y = y + 0.5f;
                    float t = static_cast<float>(y - y0) / dy_int;
                    frag.fb_position.z = start.fb_position.z + t * dz;
                    frag.attributes = va.attributes;
                    frag.derivatives.fill(Vec2(0.0f, 0.0f));
                    emit_fragment(frag);
                }
            } else {
                int decision = 2 * std::abs(dx_int) - dy_int;
                int x_step = (dx_int > 0) ? 1 : -1;
                int x = x0;
                
                for (int y = y0 + 1; y < y1; y++) {
                    Fragment frag;
                    frag.fb_position.x = x + 0.5f;
                    frag.fb_position.y = y + 0.5f;
                    float t = static_cast<float>(y - y0) / dy_int;
                    frag.fb_position.z = start.fb_position.z + t * dz;
                    frag.attributes = va.attributes;
                    frag.derivatives.fill(Vec2(0.0f, 0.0f));
                    emit_fragment(frag);
                    
                    if (decision >= 0) {
                        x += x_step;
                        decision -= 2 * dy_int;
                    }
                    decision += 2 * std::abs(dx_int);
                }
            }
        }
    } else {
        if (dx != 0) {
            int x0 = static_cast<int>(std::floor(start.fb_position.x));
            int y0 = static_cast<int>(std::floor(start.fb_position.y));
            int x1 = static_cast<int>(std::floor(end.fb_position.x));
            int y1 = static_cast<int>(std::floor(end.fb_position.y));
            
            int dx_int = x1 - x0;
            int dy_int = y1 - y0;
            int y_step = (dy_int > 0) ? 1 : -1;
            
            if (dy_int == 0) {
				for (int x = x0; x <= x1; x++) {
					//Skip the pixel containing starting and ending points (cuz they're alread dealt with!)
					if (x == x0 || x == x1) continue;
                    
                    Fragment frag;
                    frag.fb_position.x = x + 0.5f;
                    frag.fb_position.y = y0 + 0.5f;
                    float t = static_cast<float>(x - x0) / dx_int;
                    frag.fb_position.z = start.fb_position.z + t * dz;
                    frag.attributes = va.attributes;
                    frag.derivatives.fill(Vec2(0.0f, 0.0f));
                    emit_fragment(frag);
                }
            } else {
                int decision = 2 * std::abs(dy_int) - dx_int;
                int y = y0;
                
                for (int x = x0 + 1; x < x1; x++) {
                    Fragment frag;
                    frag.fb_position.x = x + 0.5f;
                    frag.fb_position.y = y + 0.5f;
                    float t = static_cast<float>(x - x0) / dx_int;
                    frag.fb_position.z = start.fb_position.z + t * dz;
                    frag.attributes = va.attributes;
                    frag.derivatives.fill(Vec2(0.0f, 0.0f));
                    emit_fragment(frag);
                    
                    if (decision >= 0) {
                        y += y_step;
                        decision -= 2 * dx_int;
                    }
                    decision += 2 * std::abs(dy_int);
                }
            }
        }
    }
}

/*
 * rasterize_triangle(a,b,c,emit) calls 'emit(frag)' at every location
 *  	(x+0.5,y+0.5) (where x,y are integers) covered by triangle (a,b,c).
 *
 * The emitted fragment should have:
 * - frag.fb_position.xy = (x+0.5, y+0.5)
 * - frag.fb_position.z = linearly interpolated fb_position.z from a,b,c (NOTE: does not depend on Interp mode!)
 * - frag.attributes = depends on Interp_* flag in flags:
 *   - if Interp_Flat: copy from va.attributes
 *   - if Interp_Smooth: interpolate as if (a,b,c) is a 2D triangle flat on the screen
 *   - if Interp_Correct: use perspective-correct interpolation
 * - frag.derivatives = derivatives w.r.t. fb_position.x and fb_position.y of the first frag.derivatives.size() attributes.
 *
 * Notes on derivatives:
 * 	The derivatives are partial derivatives w.r.t. screen locations. That is:
 *    derivatives[i].x = d/d(fb_position.x) attributes[i]
 *    derivatives[i].y = d/d(fb_position.y) attributes[i]
 *  You may compute these derivatives analytically or numerically.
 *
 *  See section 8.12.1 "Derivative Functions" of the GLSL 4.20 specification for some inspiration. (*HOWEVER*, the spec is solving a harder problem, and also nothing in the spec is binding on your implementation)
 *
 *  One approach is to rasterize blocks of four fragments and use forward and backward differences to compute derivatives.
 *  To assist you in this approach, keep in mind that the framebuffer size is *guaranteed* to be even. (see framebuffer.h)
 *
 * Notes on coverage:
 *  If two triangles are on opposite sides of the same edge, and a
 *  fragment center lies on that edge, rasterize_triangle should
 *  make sure that exactly one of the triangles emits that fragment.
 *  (Otherwise, speckles or cracks can appear in the final render.)
 * 
 *  For degenerate (co-linear) triangles, you may consider them to not be on any side of an edge.
 * 	Thus, even if two degnerate triangles share an edge that contains a fragment center, you don't need to emit it.
 *  You will not lose points for doing something reasonable when handling this case
 *
 *  This is pretty tricky to get exactly right!
 *
 */
template<PrimitiveType p, class P, uint32_t flags>
void Pipeline<p, P, flags>::rasterize_triangle(
	ClippedVertex const& va, ClippedVertex const& vb, ClippedVertex const& vc,
	std::function<void(Fragment const&)> const& emit_fragment) {
	// NOTE: it is okay to restructure this function to allow these tasks to use the
	//  same code paths. Be aware, however, that all of them need to remain working!
	//  (e.g., if you break Flat while implementing Correct, you won't get points
	//   for Flat.)
	if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Flat) {
		// A1T3: flat triangles
		// TODO: rasterize triangle (see block comment above this function).
		        // screen-space positions
        float ax = va.fb_position.x, ay = va.fb_position.y, az = va.fb_position.z;
        float bx = vb.fb_position.x, by = vb.fb_position.y, bz = vb.fb_position.z;
        float cx = vc.fb_position.x, cy = vc.fb_position.y, cz = vc.fb_position.z;

        // compute axis-aligned integer bounding box (pixel indices)
        int min_x = static_cast<int>(std::floor(std::min({ax, bx, cx})));
        int max_x = static_cast<int>(std::floor(std::max({ax, bx, cx})));
        int min_y = static_cast<int>(std::floor(std::min({ay, by, cy})));
        int max_y = static_cast<int>(std::floor(std::max({ay, by, cy})));

        // small epsilon for float comparisons
        const float EPS = 1e-6f;

        // edge function: edge(a,b,p) >0 means p is on left side of AB (given our convention)
		// AB X AP
        auto edge = [&](float ax_, float ay_, float bx_, float by_, float px, float py) -> float {
            return (px - ax_) * (by_ - ay_) - (py - ay_) * (bx_ - ax_);
        };

        // compute signed area (used to determine orientation)
        float area = edge(ax, ay, bx, by, cx, cy);
        if (std::abs(area) < EPS) {
            // degenerate triangle: nothing to rasterize
            return;
        }
        bool ccw = (area > 0.0f);

        // top-left test for each edge AB:
        auto is_top_left = [&](float ax_, float ay_, float bx_, float by_) -> bool {
            // Choose the conventional top-left rule:
            // an edge is top-left if it is a "top" edge (ay_ < by_) or if horizontal and runs left (ay_ == by_ and ax_ > bx_)
            if (ay_ < by_) return true;
            if (ay_ == by_ && ax_ > bx_) return true;
            return false;
        };

        bool tl0 = is_top_left(ax, ay, bx, by); // edge AB
        bool tl1 = is_top_left(bx, by, cx, cy); // edge BC
        bool tl2 = is_top_left(cx, cy, ax, ay); // edge CA

        // iterate pixels inside bbox
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                // find pixel center
                float sx = x + 0.5f;
                float sy = y + 0.5f;

                // compute edge values
                float w0 = edge(bx, by, cx, cy, sx, sy); // weight for vertex A
                float w1 = edge(cx, cy, ax, ay, sx, sy); // weight for vertex B
                float w2 = edge(ax, ay, bx, by, sx, sy); // weight for vertex C

                // For CCW, point is inside iff all w >= 0 with top-left tie rule
                bool pass0, pass1, pass2;
                if (ccw) {
                    pass0 = (w0 > EPS) || (std::abs(w0) <= EPS && tl0);
                    pass1 = (w1 > EPS) || (std::abs(w1) <= EPS && tl1);
                    pass2 = (w2 > EPS) || (std::abs(w2) <= EPS && tl2);
                    if (!(pass0 && pass1 && pass2)) continue;
                } else {
                    // CW: point is inside if all w <= 0
                    pass0 = (w0 < -EPS) || (std::abs(w0) <= EPS && tl0);
                    pass1 = (w1 < -EPS) || (std::abs(w1) <= EPS && tl1);
                    pass2 = (w2 < -EPS) || (std::abs(w2) <= EPS && tl2);
                    if (!(pass0 && pass1 && pass2)) continue;
                }

                // point is covered -> compute barycentric weights (using area)
                // lambda_a = edge(b,c,p) / edge(b,c,a) = w0 / area
                float inv_area = 1.0f / area;
                float lambda_a = w0 * inv_area;
                float lambda_b = w1 * inv_area;
                float lambda_c = w2 * inv_area;

                // interpolate z linearly
                float z = lambda_a * az + lambda_b * bz + lambda_c * cz;

                Fragment frag;
                frag.fb_position.x = sx;
                frag.fb_position.y = sy;
                frag.fb_position.z = z;
                frag.attributes = va.attributes;
                // derivatives are zeros for flat interpolation
                for (size_t i = 0; i < frag.derivatives.size(); ++i) {
                    frag.derivatives[i] = Vec2(0.0f, 0.0f);
                }
                emit_fragment(frag);
            }
        }
	} else if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Smooth) {
		// A1T5: screen-space smooth triangles
		// TODO: rasterize triangle (see block comment above this function).

		// As a placeholder, here's code that calls the Flat interpolation version of the function:
		//(remove this and replace it with a real solution)
		Pipeline<PrimitiveType::Lines, P, (flags & ~PipelineMask_Interp) | Pipeline_Interp_Flat>::rasterize_triangle(va, vb, vc, emit_fragment);
	} else if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Correct) {
		// A1T5: perspective correct triangles
		// TODO: rasterize triangle (block comment above this function).

		// As a placeholder, here's code that calls the Screen-space interpolation function:
		//(remove this and replace it with a real solution)
		Pipeline<PrimitiveType::Lines, P, (flags & ~PipelineMask_Interp) | Pipeline_Interp_Smooth>::rasterize_triangle(va, vb, vc, emit_fragment);
	}
}

//-------------------------------------------------------------------------
// compile instantiations for all programs and blending and testing types:

#include "programs.h"

template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Always | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Always | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Always | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Never | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Never | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Never | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Less | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Less | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Less | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Always | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Always | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Always | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Never | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Never | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Never | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Less | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Less | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Less | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Always | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Always | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Always | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Never | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Never | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Never | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Less | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Less | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Less | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Always | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Never | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Less | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Always | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Never | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Less | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Always | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Never | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Less | Pipeline_Interp_Flat>;