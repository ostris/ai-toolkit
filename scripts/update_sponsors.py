import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

# API credentials
PATREON_TOKEN = os.getenv("PATREON_ACCESS_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_ORG = os.getenv("GITHUB_ORG")  # Organization name (optional)

# Output file
README_PATH = "SUPPORTERS.md"

def fetch_patreon_supporters():
    """Fetch current Patreon supporters"""
    print("Fetching Patreon supporters...")
    
    headers = {
        "Authorization": f"Bearer {PATREON_TOKEN}",
        "Content-Type": "application/json"
    }
    
    url = "https://www.patreon.com/api/oauth2/v2/campaigns"
    
    try:
        # First get the campaign ID
        campaign_response = requests.get(url, headers=headers)
        campaign_response.raise_for_status()
        campaign_data = campaign_response.json()
        
        if not campaign_data.get('data'):
            print("No campaigns found for this Patreon account")
            return []
        
        campaign_id = campaign_data['data'][0]['id']
        
        # Now get the supporters for this campaign
        members_url = f"https://www.patreon.com/api/oauth2/v2/campaigns/{campaign_id}/members"
        params = {
            "include": "user",
            "fields[member]": "full_name,is_follower,patron_status",  # Removed profile_url
            "fields[user]": "image_url"
        }
        
        supporters = []
        while members_url:
            members_response = requests.get(members_url, headers=headers, params=params)
            members_response.raise_for_status()
            members_data = members_response.json()
            
            # Process the response to extract active patrons
            for member in members_data.get('data', []):
                attributes = member.get('attributes', {})
                
                # Only include active patrons
                if attributes.get('patron_status') == 'active_patron':
                    name = attributes.get('full_name', 'Anonymous Supporter')
                    
                    # Get user data which contains the profile image
                    user_id = member.get('relationships', {}).get('user', {}).get('data', {}).get('id')
                    profile_image = None
                    profile_url = None  # Removed profile_url since it's not supported
                    
                    if user_id:
                        for included in members_data.get('included', []):
                            if included.get('id') == user_id and included.get('type') == 'user':
                                profile_image = included.get('attributes', {}).get('image_url')
                                break
                    
                    supporters.append({
                        'name': name,
                        'profile_image': profile_image,
                        'profile_url': profile_url,  # This will be None
                        'platform': 'Patreon',
                        'amount': 0  # Placeholder, as Patreon API doesn't provide this in the current response
                    })
            
            # Handle pagination
            members_url = members_data.get('links', {}).get('next')
        
        print(f"Found {len(supporters)} active Patreon supporters")
        return supporters
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Patreon data: {e}")
        print(f"Response content: {e.response.content if hasattr(e, 'response') else 'No response content'}")
        return []
    
def fetch_github_sponsors():
    """Fetch current GitHub sponsors for a user or organization"""
    print("Fetching GitHub sponsors...")
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Determine if we're fetching for a user or an organization
    entity_type = "organization" if GITHUB_ORG else "user"
    entity_name = GITHUB_ORG if GITHUB_ORG else GITHUB_USERNAME
    
    if not entity_name:
        print("Error: Neither GITHUB_USERNAME nor GITHUB_ORG is set")
        return []
    
    # Different GraphQL query structure based on entity type
    if entity_type == "user":
        query = """
        query {
          user(login: "%s") {
            sponsorshipsAsMaintainer(first: 100) {
              nodes {
                sponsorEntity {
                  ... on User {
                    login
                    name
                    avatarUrl
                    url
                  }
                  ... on Organization {
                    login
                    name
                    avatarUrl
                    url
                  }
                }
                tier {
                  monthlyPriceInDollars
                }
                isOneTimePayment
                isActive
              }
            }
          }
        }
        """ % entity_name
    else:  # organization
        query = """
        query {
          organization(login: "%s") {
            sponsorshipsAsMaintainer(first: 100) {
              nodes {
                sponsorEntity {
                  ... on User {
                    login
                    name
                    avatarUrl
                    url
                  }
                  ... on Organization {
                    login
                    name
                    avatarUrl
                    url
                  }
                }
                tier {
                  monthlyPriceInDollars
                }
                isOneTimePayment
                isActive
              }
            }
          }
        }
        """ % entity_name
    
    try:
        response = requests.post(
            "https://api.github.com/graphql",
            headers=headers,
            json={"query": query}
        )
        response.raise_for_status()
        data = response.json()
        
        # Process the response - the path to the data differs based on entity type
        if entity_type == "user":
            sponsors_data = data.get('data', {}).get('user', {}).get('sponsorshipsAsMaintainer', {}).get('nodes', [])
        else:
            sponsors_data = data.get('data', {}).get('organization', {}).get('sponsorshipsAsMaintainer', {}).get('nodes', [])
        
        sponsors = []
        for sponsor in sponsors_data:
            # Only include active sponsors
            if sponsor.get('isActive'):
                entity = sponsor.get('sponsorEntity', {})
                name = entity.get('name') or entity.get('login', 'Anonymous Sponsor')
                profile_image = entity.get('avatarUrl')
                profile_url = entity.get('url')
                amount = sponsor.get('tier', {}).get('monthlyPriceInDollars', 0)
                
                sponsors.append({
                    'name': name,
                    'profile_image': profile_image,
                    'profile_url': profile_url,
                    'platform': 'GitHub Sponsors',
                    'amount': amount
                })
        
        print(f"Found {len(sponsors)} active GitHub sponsors for {entity_type} '{entity_name}'")
        return sponsors
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching GitHub sponsors data: {e}")
        return []

def generate_readme(supporters):
    """Generate a README.md file with supporter information"""
    print(f"Generating {README_PATH}...")
    
    # Sort supporters by amount (descending) and then by name
    supporters.sort(key=lambda x: (-x['amount'], x['name'].lower()))
    
    # Determine the proper footer links based on what's configured
    github_entity = GITHUB_ORG if GITHUB_ORG else GITHUB_USERNAME
    github_entity_type = "orgs" if GITHUB_ORG else "sponsors"
    github_sponsor_url = f"https://github.com/{github_entity_type}/{github_entity}"
    
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write("## Support My Work\n\n")
        f.write("If you enjoy my work, or use it for commercial purposes, please consider sponsoring me so I can continue to maintain it. Every bit helps! \n\n")
        # Create appropriate call-to-action based on what's configured
        cta_parts = []
        if github_entity:
            cta_parts.append(f"[Become a sponsor on GitHub]({github_sponsor_url})")
        if PATREON_TOKEN:
            cta_parts.append("[support me on Patreon](https://www.patreon.com/ostris)")
        
        if cta_parts:
            if GITHUB_ORG:
                f.write(f"{' or '.join(cta_parts)}.\n\n")
        f.write("Thank you to all my current supporters!\n\n")
        
        f.write(f"_Last updated: {datetime.now().strftime('%Y-%m-%d')}_\n\n")
        
        # Write GitHub Sponsors section
        github_sponsors = [s for s in supporters if s['platform'] == 'GitHub Sponsors']
        if github_sponsors:
            f.write("### GitHub Sponsors\n\n")
            for sponsor in github_sponsors:
                if sponsor['profile_image']:
                    f.write(f"<a href=\"{sponsor['profile_url']}\" title=\"{sponsor['name']}\"><img src=\"{sponsor['profile_image']}\" width=\"50\" height=\"50\" alt=\"{sponsor['name']}\" style=\"border-radius:50%;display:inline-block;\"></a> ")
                else:
                    f.write(f"[{sponsor['name']}]({sponsor['profile_url']}) ")
            f.write("\n\n")
        
        # Write Patreon section
        patreon_supporters = [s for s in supporters if s['platform'] == 'Patreon']
        if patreon_supporters:
            f.write("### Patreon Supporters\n\n")
            for supporter in patreon_supporters:
                if supporter['profile_image']:
                    f.write(f"<a href=\"{supporter['profile_url']}\" title=\"{supporter['name']}\"><img src=\"{supporter['profile_image']}\" width=\"50\" height=\"50\" alt=\"{supporter['name']}\" style=\"border-radius:50%;display:inline-block;\"></a> ")
                else:
                    f.write(f"[{supporter['name']}]({supporter['profile_url']}) ")
            f.write("\n\n")
        
        f.write("\n---\n\n")
        
    
    print(f"Successfully generated {README_PATH} with {len(supporters)} supporters!")

def main():
    """Main function"""
    print("Starting supporter data collection...")
    
    # Check if required environment variables are set
    missing_vars = []
    if not GITHUB_TOKEN:
        missing_vars.append("GITHUB_TOKEN")
    
    # Either username or org is required for GitHub
    if not GITHUB_USERNAME and not GITHUB_ORG:
        missing_vars.append("GITHUB_USERNAME or GITHUB_ORG")
    
    # Patreon token is optional but warn if missing
    patreon_enabled = bool(PATREON_TOKEN)
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please add them to your .env file")
        return
    
    if not patreon_enabled:
        print("Warning: PATREON_ACCESS_TOKEN not set. Will only fetch GitHub sponsors.")
    
    # Fetch data from both platforms
    patreon_supporters = fetch_patreon_supporters() if PATREON_TOKEN else []
    github_sponsors = fetch_github_sponsors()
    
    # Combine supporters from both platforms
    all_supporters = patreon_supporters + github_sponsors
    
    if not all_supporters:
        print("No supporters found on either platform")
        return
    
    # Generate README
    generate_readme(all_supporters)

if __name__ == "__main__":
    main()